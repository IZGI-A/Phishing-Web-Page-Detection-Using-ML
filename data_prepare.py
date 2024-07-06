import os
import shutil
import sys


def prepare_data():
    if len(sys.argv) != 5:
        print("Usage: python data_prepare.py <dataset_path> <phishing_folder> <benign_folder> <misleading_folder>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    phishing_data_path = os.path.join(dataset_path, sys.argv[2])
    benign_data_path = os.path.join(dataset_path, sys.argv[3])
    misleading_data_path = os.path.join(dataset_path, sys.argv[4])

    benign_misleading_path = "benign_mislead"
    phishing_path = "phishing"

    os.makedirs(benign_misleading_path, exist_ok=True)
    os.makedirs(phishing_path, exist_ok=True)

    counter = 0
    phishing = 0
    benign_mislead = 0

    def copy_html_files(source_path, target_path):
        nonlocal counter, total, benign_mislead, phishing
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.endswith("html.txt"):
                    relative_path = os.path.relpath(root, source_path).replace('.', '_')
                    dest_file_path = os.path.join(target_path, f"{relative_path}_{file}")

                    shutil.copy(os.path.join(root, file), dest_file_path)
                    counter += 1
                    total += 1
                    if source_path in [benign_data_path, misleading_data_path]:
                        benign_mislead += 1
                    elif source_path == phishing_data_path:
                        phishing += 1

    copy_html_files(benign_data_path, benign_misleading_path)
    copy_html_files(misleading_data_path, benign_misleading_path)
    copy_html_files(phishing_data_path, phishing_path)

    print(f"Total files processed: {counter}")
    print(f"Phishing files: {phishing}")
    print(f"Benign and Misleading files: {benign_mislead}")


if __name__ == '__main__':
    prepare_data()
