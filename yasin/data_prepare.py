import multiprocessing
import os
import re
import shutil
import time
import zipfile
from functools import partial


def unzip_file(source_path, target_path, delete: bool):
    zipfile.ZipFile(source_path, 'r').extractall(target_path)
    if delete:
        os.remove(source_path)
    return None


def change_directory_names(path):
    files = os.listdir(path)
    for i in files:
        old_path = path + "/" + i
        new_path = path + "/" + re.sub(r'[^0-9a-zA-Z]+', '_', i)
        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
    return None


def safe_copy_paste(file_name, source_path, target_path):
    html_path = source_path + "/" + file_name + "/html.txt"
    file_name = re.sub(r'[^0-9a-zA-Z]+', '_', file_name)
    if not os.path.exists(html_path):
        return False
    target_path = target_path + "/" + file_name + ".txt"
    shutil.copy2(html_path, target_path)
    return True


def copy_paste_html_contents(source_path: str, target_path: str):
    pool = multiprocessing.Pool(processes=os.cpu_count())
    partial_safe_copy_paste = partial(safe_copy_paste, source_path=source_path, target_path=target_path)
    pool.map(partial_safe_copy_paste, os.listdir(source_path))
    pool.close()
    return None


def extract_htmls(source_path, target_path, unzip: bool, delete_zip: bool):
    # You can run this part to unzip files if you do not extract files.
    # If you get FileNotFoundError because of too long path, you can follow these instructions:
    # 1. Type regedit in the start menu and search
    # 2. Navigate to Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
    # 3. Edit LongPathsEnabled ( set value from 0 to 1)
    if unzip:
        unzip_file(source_path + ".zip", source_path, delete_zip)
    change_directory_names(source_path)
    copy_paste_html_contents(source_path, target_path)
    return None


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def main():
    source_path = "PhishIntention"
    target_path = "PreparedData"

    benign_source_path = source_path + "/" + "benign_25k"
    misleading_source_path = source_path + "/" + "misleading"
    phishing_source_path = source_path + "/" + "phish_sample_30k"

    legitimate_target_path = target_path + "/Legitimate"
    phishing_target_path = target_path + "/Phishing"

    check_path(legitimate_target_path)
    check_path(phishing_target_path)

    print("Extracting HTML files. This may take a while...")
    start_time = time.time()
    extract_htmls(benign_source_path, legitimate_target_path, True, False)
    extract_htmls(misleading_source_path, legitimate_target_path, True, False)
    extract_htmls(phishing_source_path, phishing_target_path, True, False)
    print("Extracted all HTML files in {} second.".format(time.time() - start_time))


if __name__ == '__main__':
    main()
