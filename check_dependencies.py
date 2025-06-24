#!/usr/bin/env python3
"""
Script to check current versions of dependencies listed in requirements.txt files.
Compares specified versions in requirements.txt with actually installed versions.
Can also update requirements.txt files with actual installed versions.
"""

import os
import re
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def find_requirements_files(repo_path: str) -> List[Path]:
    """Find all requirements.txt files in the repository."""
    requirements_files = []
    for root, dirs, files in os.walk(repo_path):
        #Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            if file == 'requirements.txt':
                requirements_files.append(Path(root) / file)
    
    return requirements_files


def parse_requirements_file(file_path: Path) -> List[Tuple[str, str, int]]:
    """Parse a requirements.txt file and extract package names, versions, and line numbers."""
    packages = []
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                original_line = line.rstrip('\n')
                line = line.strip()
                
                #Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                #Parse package specification
                #Handle formats like: package, package==version, package>=version, etc.
                match = re.match(r'^([a-zA-Z0-9_-]+)(.*)$', line)
                if match:
                    package_name = match.group(1)
                    version_spec = match.group(2).strip()
                    packages.append((package_name, version_spec, line_num, original_line))
                else:
                    print(f"Warning: Could not parse line {line_num} in {file_path}: {line}")
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return packages


def get_installed_version(package_name: str) -> Optional[str]:
    """Get the currently installed version of a package."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            #Parse the output to extract version
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
        
        return None
    
    except Exception as e:
        print(f"Error checking version for {package_name}: {e}")
        return None


def update_requirements_file(file_path: Path, installed_versions: Dict[str, str]) -> bool:
    """Update a requirements.txt file with actual installed versions."""
    try:
        #Read the entire file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        #Parse packages to get line numbers
        packages = parse_requirements_file(file_path)
        updated_lines = lines.copy()
        
        #Track which packages were updated
        updated_packages = []
        
        for package_name, version_spec, line_num, original_line in packages:
            if package_name in installed_versions:
                installed_version = installed_versions[package_name]
                if installed_version:
                    #Create new line with exact version
                    new_line = f"{package_name}=={installed_version}\n"
                    updated_lines[line_num - 1] = new_line
                    updated_packages.append(package_name)
        
        #Write back to file
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)
        
        return True, updated_packages
    
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False, []


def check_dependencies():
    """Main function to check all dependencies."""
    repo_path = Path.cwd()
    print(f"Checking dependencies in repository: {repo_path}")
    print("=" * 80)
    
    #Find all requirements.txt files
    requirements_files = find_requirements_files(repo_path)
    
    if not requirements_files:
        print("No requirements.txt files found in the repository.")
        return
    
    print(f"Found {len(requirements_files)} requirements.txt file(s):")
    for req_file in requirements_files:
        print(f"  - {req_file.relative_to(repo_path)}")
    print()
    
    #Collect all unique packages
    all_packages = set()
    file_packages = {}
    
    for req_file in requirements_files:
        packages = parse_requirements_file(req_file)
        file_packages[req_file] = packages
        
        for package_name, version_spec, line_num, original_line in packages:
            all_packages.add(package_name)
    
    #Get installed versions for all packages
    installed_versions = {}
    for package_name in sorted(all_packages):
        installed_version = get_installed_version(package_name)
        installed_versions[package_name] = installed_version
    
    #Display results
    print("DEPENDENCY VERSION CHECK RESULTS")
    print("=" * 80)
    
    for req_file in requirements_files:
        print(f"\n📁 {req_file.relative_to(repo_path)}:")
        print("-" * 60)
        
        packages = file_packages[req_file]
        if not packages:
            print("  No packages found in this file.")
            continue
        
        for package_name, version_spec, line_num, original_line in packages:
            installed_version = installed_versions[package_name]
            
            if installed_version:
                status = "✅" if version_spec == "" or check_version_compatibility(installed_version, version_spec) else "⚠️"
                print(f"  {status} {package_name}")
                print(f"      Specified: {version_spec if version_spec else 'any'}")
                print(f"      Installed:  {installed_version}")
            else:
                print(f"  ❌ {package_name}")
                print(f"      Specified: {version_spec if version_spec else 'any'}")
                print(f"      Installed:  NOT INSTALLED")
            print()
    
    #Summary
    print("SUMMARY")
    print("=" * 80)
    total_packages = len(all_packages)
    installed_count = sum(1 for v in installed_versions.values() if v is not None)
    missing_count = total_packages - installed_count
    
    print(f"Total unique packages: {total_packages}")
    print(f"Installed packages: {installed_count}")
    print(f"Missing packages: {missing_count}")
    
    if missing_count > 0:
        print("\nMissing packages:")
        for package_name, version in installed_versions.items():
            if version is None:
                print(f"  - {package_name}")
    
    return installed_versions


def update_all_requirements_files(installed_versions: Dict[str, str]):
    """Update all requirements.txt files with actual installed versions."""
    repo_path = Path.cwd()
    requirements_files = find_requirements_files(repo_path)
    
    if not requirements_files:
        print("No requirements.txt files found to update.")
        return
    
    print("\n🔄 UPDATING REQUIREMENTS.TXT FILES")
    print("=" * 80)
    
    total_updated = 0
    
    for req_file in requirements_files:
        print(f"\n📁 Updating {req_file.relative_to(repo_path)}:")
        
        success, updated_packages = update_requirements_file(req_file, installed_versions)
        
        if success:
            if updated_packages:
                print(f"  ✅ Updated {len(updated_packages)} package(s):")
                for package in updated_packages:
                    print(f"     - {package}=={installed_versions[package]}")
                total_updated += len(updated_packages)
            else:
                print("  ℹ️  No packages needed updating")
        else:
            print(f"  ❌ Failed to update file")
    
    print(f"\n📊 SUMMARY: Updated {total_updated} package specifications across {len(requirements_files)} file(s)")


def check_version_compatibility(installed_version: str, version_spec: str) -> bool:
    """Check if installed version satisfies the version specification."""
    if not version_spec:
        return True
    
    try:
        from packaging import version as pkg_version
        from packaging.specifiers import SpecifierSet
        
        spec = SpecifierSet(version_spec)
        return pkg_version.parse(installed_version) in spec
    
    except ImportError:
        #Fallback: simple string comparison for common cases
        if version_spec.startswith('=='):
            return installed_version == version_spec[2:]
        elif version_spec.startswith('>='):
            #Simple version comparison (not perfect but better than nothing)
            return installed_version >= version_spec[2:]
        else:
            return True


def main():
    """Main entry point with command line argument support."""
    parser = argparse.ArgumentParser(
        description="Check and optionally update requirements.txt files with installed versions"
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update requirements.txt files with actual installed versions'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check versions without updating (default behavior)'
    )
    
    args = parser.parse_args()
    
    #Check dependencies first
    installed_versions = check_dependencies()
    
    #Update if requested
    if args.update:
        print("\n" + "=" * 80)
        response = input("Do you want to update all requirements.txt files with actual installed versions? (y/N): ")
        if response.lower() in ['y', 'yes']:
            update_all_requirements_files(installed_versions)
        else:
            print("Update cancelled.")


if __name__ == "__main__":
    main() 