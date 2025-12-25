---
name: pdf-processing
description: Extract text and tables from PDF files, fill forms, merge documents. Use when the user needs to work with PDF files.
license: MIT
compatibility: Requires pdfplumber and PyPDF2 packages
metadata:
  author: test-org
  version: "1.0"
---

# PDF Processing

## When to use this skill
Use this skill when the user needs to work with PDF files, including:
- Extracting text from PDFs
- Extracting tables from PDFs
- Filling PDF forms
- Merging multiple PDFs

## How to extract text
1. Use pdfplumber for text extraction
2. Open the PDF file using `pdfplumber.open(file_path)`
3. Iterate through pages and extract text

## How to fill forms
1. Use PyPDF2 to read the PDF
2. Access form fields
3. Fill in the required values
