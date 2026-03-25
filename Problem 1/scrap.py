import requests
from bs4 import BeautifulSoup
import pdfplumber
import re

# Source URLs
source_urls = [
    # Official Institute Pages
    "https://iitj.ac.in/main/en/introduction",
    "https://iitj.ac.in/main/en/iitj",
    "https://iitj.ac.in/main/en/director",
    "https://iitj.ac.in/main/en/chairman",
    "https://iitj.ac.in/main/en/why-pursue-a-career-@-iit-jodhpur",
    "https://iitj.ac.in/office-of-students/en/campus-life",
    "https://iitj.ac.in/office-of-students/en/office-of-students",
    "https://iitj.ac.in/main/en/news",
    "https://iitj.ac.in/main/en/events",
    "https://iitj.ac.in/Main/en/Annual-Reports-of-the-Institute",

    # Academic Regulations & Programs
    "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    "https://iitj.ac.in/office-of-academics/en/circulars",
    "https://iitj.ac.in/office-of-academics/en/curriculum",
    "https://iitj.ac.in/office-of-academics/en/program-Structure",
    "https://iitj.ac.in/office-of-academics/en/list-of-academic-programs",
    "https://iitj.ac.in/office-of-academics/en/b.tech.",
    "https://iitj.ac.in/office-of-academics/en/m.tech.",
    "https://iitj.ac.in/office-of-academics/en/ph.d.",
    "https://iitj.ac.in/office-of-academics/en/mba",
    "https://iitj.ac.in/itep/",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/btech",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/mtech",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/courses",
    "https://iitj.ac.in/office-of-academics/en/academics",
    "https://iitj.ac.in/Office-of-Academics/en/Academic-Calendar",
    "https://iitj.ac.in/office-of-academics/en/ug-registration-guidelines",
    "https://iitj.ac.in/admission-postgraduate-programs/en/Admission-to-Postgraduate-Programs",
    "https://iitj.ac.in/office-of-executive-education/en/office-of-executive-education",
    "https://iitj.ac.in/office-of-academics/en/scholarships",
    "https://iitj.ac.in/office-of-academics/en/convocation",
    "https://iitj.ac.in/main/en/faqs-applicants",
    "https://iitj.ac.in/office-of-registrar/en/office-of-registrar",
    "https://iitj.ac.in/office-of-administration/en/office-of-administration",

    # Newsletters
    "https://iitj.ac.in/institute-repository/en/Newsletter",

    # Departments & Centers
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/school-of-artificial-intelligence-and-data-science",
    "https://iitj.ac.in/es/en/engineering-science",
    "https://iitj.ac.in/school-of-liberal-arts/",
    "https://iitj.ac.in/school-of-design/",
    "https://iitj.ac.in/schools/",
    "https://iitj.ac.in/m/Index/main-departments?lg=en",
    "https://iitj.ac.in/m/Index/main-centers?lg=en",
    "https://iitj.ac.in/m/Index/main-idrps-idrcs?lg=en",
    "https://iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    "https://iitj.ac.in/main/en/research-highlight",
    "https://iitj.ac.in/crf/en/crf",
    "https://iitj.ac.in/techscape/en/Techscape",
    "https://iitj.ac.in/health-center/en/health-center",

    # Faculty Profiles
    "https://iitj.ac.in/main/en/faculty-members",
    "https://iitj.ac.in/main/en/visiting-faculty-members",
    "https://iitj.ac.in/main/en/scholars-in-residence",
    "https://iitj.ac.in/People/List?dept=computer-science-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=electrical-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=mechanical-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=physics&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=mathematics&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=chemistry&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=bioscience-bioengineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=civil-and-infrastructure-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=metallurgical-and-materials-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=school-of-artificial-intelligence-data-science&c=ce26246f-00c9-4286-bb4c-7f023b4c5460",
    "https://iitj.ac.in/People/List?dept=school-of-liberal-arts&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=school-of-design&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=schools&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"
]

pdf_files = [
    "https://iitj.ac.in/PageImages/Gallery/03-2025/4_Regulation_PG_2022-onwards_20022023.pdf" 
]

def scrape_page(url):
    # standard headers to avoid getting blocked by the server
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # removing tags like script, style etc. to extract main text from the page
        bad_tags = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'meta', 'button']
        for tag in soup.find_all(bad_tags):
            tag.decompose() 
            
        # using space separator so words from different blocks don't get mashed together
        return soup.get_text(separator=' ', strip=True)
        
    except Exception as e:
        print(f"--> fetch failed for {url}: {e}")
        return ""

def extract_pdf_text(filepath):
    # Using page by page extraction for PDFs
    extracted_pages = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_pages.append(text)
        return " ".join(extracted_pages)
    except Exception as e:
        print(f"--> PDF read error ({filepath}): {e}")
        return ""

if __name__ == "__main__":
    all_text = []
    
    print("Starting data collection...")
    
    # 1. Get HTML data
    for link in source_urls:
        print(f"Fetching: {link}")
        page_text = scrape_page(link)
        
        if page_text:
            # A simple check to make sure that we actually got meaningful content from the link
            if len(page_text) < 100:
                print(f"Note: Got very little text from {link}. Check the div class.")
            all_text.append(page_text)
            
    print(f"Scraped {len(all_text)} web pages so far.")
            
    # 2. Get PDF data
    for doc in pdf_files:
        print(f"Reading PDF: {doc}")
        pdf_data = extract_pdf_text(doc)
        if pdf_data:
            all_text.append(pdf_data)
            
    # Joined everything together
    combined_text = " ".join(all_text)
    
    # Cleaning whitespaces before the actual NLP task
    cleaned_dataset = re.sub(r'\s+', ' ', combined_text).strip()
    
    if cleaned_dataset:
        output_name = "Problem 1/iitj_corpus.txt"
        with open(output_name, "w", encoding="utf-8") as f:
            f.write(cleaned_dataset)
        print(f"\nSuccess! Total dataset length: {len(cleaned_dataset)} chars.")
        print(f"Saved to {output_name}")
    else:
        print("\nScript has executed successfully but no data was collected. Check URLs.")