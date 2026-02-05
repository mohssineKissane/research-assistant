"""
Script to create a sample PDF file for testing the DocumentLoader
"""
from pathlib import Path

def create_sample_pdf():
    """Create a simple sample PDF using reportlab"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        # Define the output path
        output_dir = Path(__file__).parent / "data" / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "sample.pdf"
        
        # Create PDF
        c = canvas.Canvas(str(output_file), pagesize=letter)
        
        # Page 1
        c.setFont("Helvetica-Bold", 24)
        c.drawString(100, 750, "Sample Research Document")
        c.setFont("Helvetica", 12)
        c.drawString(100, 700, "This is a test PDF for the Research Assistant project.")
        c.drawString(100, 680, "")
        c.drawString(100, 660, "Introduction:")
        c.drawString(100, 640, "This document demonstrates the PDF loading capabilities")
        c.drawString(100, 620, "of the DocumentLoader class using LangChain and PyPDF.")
        c.showPage()
        
        # Page 2
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Page 2: Technical Details")
        c.setFont("Helvetica", 12)
        c.drawString(100, 700, "The DocumentLoader extracts text from each page")
        c.drawString(100, 680, "and creates Document objects with metadata.")
        c.drawString(100, 660, "")
        c.drawString(100, 640, "Metadata includes:")
        c.drawString(120, 620, "- Filename")
        c.drawString(120, 600, "- Upload date")
        c.drawString(120, 580, "- Page number")
        c.showPage()
        
        # Page 3
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Page 3: Conclusion")
        c.setFont("Helvetica", 12)
        c.drawString(100, 700, "This sample PDF can be used to test:")
        c.drawString(120, 680, "1. PDF loading functionality")
        c.drawString(120, 660, "2. Text extraction")
        c.drawString(120, 640, "3. Metadata handling")
        c.drawString(120, 620, "4. Multi-page document processing")
        c.showPage()
        
        c.save()
        
        print(f"✅ Successfully created sample PDF at: {output_file}")
        print(f"📄 File size: {output_file.stat().st_size} bytes")
        return str(output_file)
        
    except ImportError:
        print("❌ reportlab is not installed.")
        print("\nTo install it, run:")
        print("  pip install reportlab")
        print("\nOR")
        print("\nManually copy any PDF file to:")
        print(f"  {Path(__file__).parent / 'data' / 'samples' / 'sample.pdf'}")
        return None

if __name__ == "__main__":
    create_sample_pdf()
