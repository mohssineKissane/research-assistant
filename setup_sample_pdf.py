"""
Simple guide to add a sample PDF for testing
"""
from pathlib import Path

def main():
    # Define the target directory
    target_dir = Path(__file__).parent / "data" / "samples"
    target_file = target_dir / "sample.pdf"
    
    print("=" * 60)
    print("SAMPLE PDF SETUP GUIDE")
    print("=" * 60)
    print()
    print("The notebook needs a PDF file at:")
    print(f"  {target_file}")
    print()
    print("OPTIONS TO ADD A SAMPLE PDF:")
    print()
    print("Option 1: Copy an existing PDF")
    print("  - Find any PDF file on your computer")
    print("  - Copy it to the folder above")
    print("  - Rename it to 'sample.pdf'")
    print()
    print("Option 2: Download a sample research paper")
    print("  - Visit: https://arxiv.org/")
    print("  - Download any paper (click 'PDF' button)")
    print("  - Save it as 'sample.pdf' in the folder above")
    print()
    print("Option 3: Create a simple text file and convert to PDF")
    print("  - Open Microsoft Word or Google Docs")
    print("  - Write some sample text")
    print("  - Save/Export as PDF to the folder above")
    print()
    print("=" * 60)
    print()
    
    # Check if file already exists
    if target_file.exists():
        print(f"SUCCESS: PDF file already exists!")
        print(f"Location: {target_file}")
        print(f"Size: {target_file.stat().st_size:,} bytes")
    else:
        print(f"WAITING: No PDF file found yet.")
        print(f"Please add a PDF file to: {target_file}")

if __name__ == "__main__":
    main()
