#!/usr/bin/env python3
"""
VarFreqAnalyzer - VCF Variant Frequency Tool with Internal Variant Frequency Summary

Features:
 - Scans VCF files in a directory (and all subdirectories)
 - Extracts variants in a genomic coordinate range
 - Calculates allele frequencies per VCF
 - Optional REF and ALT input for internal variant frequency
 - Displays:
     1. All variant results (per file)
     2. Internal variant frequency summary (if REF & ALT given)
 - PyQt5 GUI
"""

import os
import sys
import re
import gzip
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal
import matplotlib.pyplot as plt

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")


# ---------------- VCF Parser Utility ----------------
class VCFParserUtil:
    @staticmethod
    def get_vcf_reader(filepath):
        try:
            if filepath.endswith(".gz"):
                return gzip.open(filepath, 'rt', encoding='utf-8')
            else:
                return open(filepath, 'r', encoding='utf-8')
        except Exception as e:
            logging.error(f"Failed to open VCF file {filepath}: {e}")
            return None

    @staticmethod
    def parse_genotype(gt_field):
        if not gt_field or ('.' in gt_field and len(gt_field) < 3):
            return None, None
        alleles = re.split(r'[|/]', gt_field)
        a1 = int(alleles[0]) if alleles[0].isdigit() else None
        a2 = int(alleles[1]) if len(alleles) > 1 and alleles[1].isdigit() else None
        return a1, a2


# ---------------- Database Manager ----------------
class DatabaseManager:
    """
    Manages VCF file discovery and repository structure.
    Recursively scans directories to locate all .vcf and .vcf.gz files.
    """
    def __init__(self, repository_path):
        self.repository_path = repository_path
        self.vcf_files = self.scan_vcf_files()

    def scan_vcf_files(self):
        """Recursively scan the repository (including subdirectories) for .vcf or .vcf.gz files."""
        vcf_files = []
        if not os.path.exists(self.repository_path):
            logging.error("Repository directory does not exist: %s", self.repository_path)
            return vcf_files

        # Handle case where the provided path is a single VCF file
        if os.path.isfile(self.repository_path):
            if self.repository_path.endswith(".vcf") or self.repository_path.endswith(".vcf.gz"):
                logging.warning("Repository path is a single VCF file. Analyzing just this file.")
                vcf_files.append(self.repository_path)
                return vcf_files
            else:
                logging.error("Repository path is an invalid file: %s", self.repository_path)
                return vcf_files

        # Recursively walk through directories and subdirectories
        for root, _, files in os.walk(self.repository_path):
            for filename in files:
                if filename.endswith(".vcf") or filename.endswith(".vcf.gz"):
                    vcf_files.append(os.path.join(root, filename))

        logging.info("Found %d VCF file(s) in %s (including subdirectories).", len(vcf_files), self.repository_path)
        return vcf_files


# ---------------- Variant Processor ----------------
class VariantProcessor:
    def __init__(self, vcf_files):
        self.vcf_files = vcf_files

    def parse_coordinate(self, coord_str):
        pattern = r"([\w]+):(\d+)(?:-(\d+))?"
        match = re.match(pattern, coord_str)
        if not match:
            raise ValueError("Invalid coordinate format. Use: chr1:1000000-2000000 or 1:1000000")
        chrom = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) is not None else start
        return chrom, start, end

    @staticmethod
    def process_file(vcf_file, target_chrom, target_start, target_end):
        file_results = {"filename": os.path.basename(vcf_file), "variants": []}
        reader = VCFParserUtil.get_vcf_reader(vcf_file)
        if reader is None:
            return file_results
        try:
            header_line = None
            for line in reader:
                if line.startswith('##'):
                    continue
                if line.startswith('#CHROM'):
                    header_line = line.strip().split('\t')
                    break
            if not header_line:
                return file_results

            for line in reader:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                chrom = parts[0]
                pos = int(parts[1])
                ref = parts[3]
                alts = parts[4].split(',')
                if chrom != target_chrom or pos < target_start or pos > target_end:
                    continue
                fmt_fields = parts[8].split(':')
                try:
                    gt_index = fmt_fields.index("GT")
                except ValueError:
                    continue
                total_alleles = 0
                alt_count = 0
                for sample_data in parts[9:]:
                    sample_info = sample_data.split(':')
                    if len(sample_info) > gt_index:
                        gt_str = sample_info[gt_index]
                        a1, a2 = VCFParserUtil.parse_genotype(gt_str)
                        if a1 is not None and a2 is not None:
                            total_alleles += 2
                            alt_count += (a1 > 0) + (a2 > 0)
                        elif a1 is not None or a2 is not None:
                            total_alleles += 1
                            alt_count += (1 if (a1 or a2) and (a1 or a2) > 0 else 0)
                af = alt_count / total_alleles if total_alleles > 0 else None
                file_results["variants"].append({
                    "pos": pos,
                    "ref": ref,
                    "alts": alts,
                    "alt_count": alt_count,
                    "total_alleles": total_alleles,
                    "allele_frequency": af
                })
        except Exception as e:
            logging.error(f"Error processing {vcf_file}: {e}")
        finally:
            reader.close()
        return file_results

    def process_coordinate(self, coord_str):
        chrom, start, end = self.parse_coordinate(coord_str)
        results = []
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.process_file, vcf, chrom, start, end): vcf for vcf in self.vcf_files}
            for f in as_completed(futures):
                results.append(f.result())
        return results


# ---------------- Worker Thread ----------------
class AnalysisWorker(QThread):
    finishedSignal = pyqtSignal(list)
    errorSignal = pyqtSignal(str)

    def __init__(self, vcf_files, coord_str):
        super().__init__()
        self.vcf_files = vcf_files
        self.coord_str = coord_str

    def run(self):
        try:
            processor = VariantProcessor(self.vcf_files)
            results = processor.process_coordinate(self.coord_str)
            self.finishedSignal.emit(results)
        except Exception as e:
            self.errorSignal.emit(str(e))
            self.finishedSignal.emit([])


# ---------------- GUI Controller ----------------
class GUIController(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VarFreqAnalyzer - VCF Population Frequency Tool")
        self.setGeometry(150, 150, 950, 750)
        self.repository_path = ""
        self.db_manager = None
        self.results = []
        self.worker = None
        self.init_ui()

    def init_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Repository path
        repo_layout = QtWidgets.QHBoxLayout()
        repo_layout.addWidget(QtWidgets.QLabel("VCF Repository Path:*"))
        self.repo_input = QtWidgets.QLineEdit()
        self.repo_browse = QtWidgets.QPushButton("Browse")
        self.repo_browse.clicked.connect(self.browse_repository)
        repo_layout.addWidget(self.repo_input)
        repo_layout.addWidget(self.repo_browse)
        layout.addLayout(repo_layout)

        # Coordinates
        coord_layout = QtWidgets.QHBoxLayout()
        coord_layout.addWidget(QtWidgets.QLabel("Coordinates* (e.g., chr1:10000-20000):"))
        self.coord_input = QtWidgets.QLineEdit()
        coord_layout.addWidget(self.coord_input)
        layout.addLayout(coord_layout)

        # REF/ALT input
        refalt_layout = QtWidgets.QHBoxLayout()
        refalt_layout.addWidget(QtWidgets.QLabel("REF and ALT (e.g., A G):"))
        self.ref_input = QtWidgets.QLineEdit()
        self.ref_input.setPlaceholderText("REF")
        self.alt_input = QtWidgets.QLineEdit()
        self.alt_input.setPlaceholderText("ALT")
        refalt_layout.addWidget(self.ref_input)
        refalt_layout.addWidget(self.alt_input)
        layout.addLayout(refalt_layout)

        # Buttons
        self.run_button = QtWidgets.QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.start_analysis)
        layout.addWidget(self.run_button)

        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        self.plot_button = QtWidgets.QPushButton("Plot Allele Frequency Distribution")
        self.plot_button.clicked.connect(self.plot_distribution)
        self.plot_button.setEnabled(False)
        layout.addWidget(self.plot_button)

        central.setLayout(layout)
        self.setCentralWidget(central)

    def browse_repository(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select VCF Repository")
        if directory:
            self.repository_path = directory
            self.repo_input.setText(directory)
            self.db_manager = DatabaseManager(directory)
            self.results_text.append(f"Found {len(self.db_manager.vcf_files)} VCF file(s) (including subdirectories).")

    def start_analysis(self):
        coord = self.coord_input.text().strip()
        repo = self.repo_input.text().strip()
        if not repo or not coord:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please select a repository and enter coordinates.")
            return
        if not os.path.isdir(repo):
            QtWidgets.QMessageBox.critical(self, "Error", f"Invalid directory: {repo}")
            return
        if self.db_manager is None:
            self.db_manager = DatabaseManager(repo)

        self.results_text.clear()
        self.results_text.append("Analyzing... please wait.")
        self.run_button.setEnabled(False)
        self.worker = AnalysisWorker(self.db_manager.vcf_files, coord)
        self.worker.finishedSignal.connect(self.analysis_finished)
        self.worker.errorSignal.connect(self.display_error)
        self.worker.start()

    def display_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Error", msg)
        self.results_text.append(f"ERROR: {msg}")
        self.run_button.setEnabled(True)

    def analysis_finished(self, results):
        self.results = results
        self.display_results()
        self.run_button.setEnabled(True)
        has_data = any(len(f["variants"]) > 0 for f in results)
        self.plot_button.setEnabled(has_data)

    def display_results(self):
        """Display all variants, then internal variant frequency summary."""
        self.results_text.clear()
        if not self.results:
            self.results_text.append("No results found.")
            return

        total_files = len(self.results)
        ref = self.ref_input.text().strip()
        alt = self.alt_input.text().strip()
        matched_variants = []

        # Show all variant results first
        self.results_text.append("Status: Analysis Complete.\n")
        for file_res in self.results:
            fname = file_res["filename"]
            if not file_res["variants"]:
                continue

            self.results_text.append(f"File: {fname}")
            for var in file_res["variants"]:
                af = var['allele_frequency']
                af_str = f"{af:.3f}" if af is not None else "NA"
                self.results_text.append(
                    f"  POS: {var['pos']} | REF: {var['ref']} | ALTS: {','.join(var['alts'])} | AF: {af_str}"
                )

                if ref and alt and var['ref'] == ref and alt in var['alts']:
                    matched_variants.append((fname, var))
            self.results_text.append("-" * 60)

        # Internal variant frequency summary (if REF & ALT provided)
        if ref and alt:
            internal_count = len({f for f, _ in matched_variants})
            coord_str = self.coord_input.text().strip()
            self.results_text.append("\n" + "=" * 60)
            self.results_text.append(
                f"Internal variant frequency for {coord_str} REF={ref} ALT={alt} = "
                f"{internal_count} out of {total_files} files"
            )
            self.results_text.append("-" * 60)

            if matched_variants:
                for f, var in matched_variants:
                    af_str = f"{var['allele_frequency']:.3f}" if var['allele_frequency'] is not None else "NA"
                    self.results_text.append(
                        f"Sample Name: {f} | POS: {var['pos']} | REF: {var['ref']} | "
                        f"ALTS: {','.join(var['alts'])} | AF: {af_str}"
                    )
            else:
                self.results_text.append("No matching variants found for given REF and ALT.")
            self.results_text.append("=" * 60)

    def plot_distribution(self):
        allele_freqs = [v['allele_frequency'] for f in self.results for v in f['variants'] if v['allele_frequency'] is not None]
        if not allele_freqs:
            QtWidgets.QMessageBox.information(self, "Info", "No allele frequency data to plot.")
            return
        plt.figure(figsize=(8, 6))
        n_bins = int(np.sqrt(len(allele_freqs))) if len(allele_freqs) > 10 else 10
        plt.hist(allele_freqs, bins=n_bins, color="#4c72b0", edgecolor="black", alpha=0.8)
        plt.xlabel("Allele Frequency (AF)")
        plt.ylabel("Count")
        plt.title("Allele Frequency Distribution")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


# ---------------- Main ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
    gui = GUIController()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
