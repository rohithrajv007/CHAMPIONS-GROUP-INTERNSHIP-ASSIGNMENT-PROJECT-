import csv
import random

company_names = [
    "Google", "Microsoft", "Amazon", "Apple", "Meta", "IBM", "Intel", "Salesforce", "Oracle", "SAP",
    "Adobe", "Tesla", "Netflix", "Uber", "Airbnb", "SpaceX", "Spotify", "PayPal", "Twitter", "Stripe",
    "NVIDIA", "Qualcomm", "Cisco", "LinkedIn", "Dell", "HP", "Zoom", "Square", "Red Hat", "VMware",
    "Accenture", "TCS", "Infosys", "Wipro", "Capgemini", "Cognizant", "Dropbox", "Slack", "Pinterest", "Snap",
    "GitHub", "GitLab", "Bitbucket", "Atlassian", "Cloudflare", "MongoDB", "Databricks", "Palantir", "Okta", "ServiceNow",
    "Workday", "Twilio", "DocuSign", "Shopify", "Etsy", "DoorDash", "Gojek", "Grab", "Flipkart", "Xiaomi",
    "Huawei", "Samsung", "LG", "Sony", "Ericsson", "Seagate", "Western Digital", "Acer", "Asus", "Lenovo",
    "Yandex", "Baidu", "Tencent", "Alibaba", "JD.com", "Rakuten", "Naver", "ByteDance", "Meituan", "Didi",
    "Robinhood", "Coinbase", "Binance", "Plaid", "SoFi", "Stripe", "Square", "Ant Financial", "Revolut", "Wise",
    "Expedia", "Booking.com", "TripAdvisor", "KAYAK", "Trivago", "Slack", "Zendesk", "Freshworks", "Hootsuite", "Buffer",
    "Mailchimp", "HubSpot", "Marketo", "Sprout Social", "Intercom", "SurveyMonkey", "Typeform", "Calendly", "ZoomInfo", "Morningstar",
    "Bloomberg", "Reuters", "S&P Global", "Moody's", "CB Insights", "Crunchbase", "PitchBook", "OpenAI", "DeepMind", "Anthropic"
]

def rand_post():
    posts = [
        "Announcing new AI-powered features for better customer engagement.",
        "Sustainability initiatives: Our carbon-neutral goal for 2026.",
        "Now hiring talented engineers worldwide!",
        "Our founder will speak at Tech World Expo next week.",
        "Product launch: Secure Cloud Data Platform now available.",
        "Awarded 'Best Employer' in tech sector for 2025."
    ]
    return random.choice(posts)

def rand_job():
    jobs = [
        "Software Engineer (Remote)", "Data Scientist (San Francisco)", "Product Manager (London)", 
        "UX Designer (New York)", "Marketing Lead (Berlin)", "", "", "DevOps Engineer (Toronto)"
    ]
    return random.choice(jobs)

def rand_people():
    counts = [f"{random.randint(500, 200000)} employees", "Data not available", "Over 10,000 staff worldwide"]
    return random.choice(counts)

about_samples = [
    "A global leader in cloud and AI, empowering innovation through cutting-edge platforms and expert services.",
    "Dedicated to transforming industries with digital solutions in analytics, AI, and cyber security.",
    "Delivering seamless digital experiences with a customer-centric approach in software, hardware, and cloud.",
    "Committed to sustainable growth and technology-driven transformation for businesses of all sizes.",
    "Transforming communication and collaboration through next-gen technologies in mobility and cloud."
]

insights_samples = [
    "Ranked Top 10 for patent filings in AI technology (2025).",
    "Featured in Fortune 100 companies driving digital transformation.",
    "Achieved $10B revenue milestone last quarter.",
    "Recognized for best-in-class workplace diversity and inclusion.",
    "Selected by UN for global AI ethics initiative."
]

with open('linkedin_companies_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Company', 'Home', 'About', 'Posts', 'Jobs', 'People', 'Insights'])
    for company in company_names:
        home = f"{company} - Innovate. Transform. Lead."
        about = random.choice(about_samples)
        posts = rand_post()
        jobs = rand_job()
        people = rand_people()
        insights = random.choice(insights_samples)
        writer.writerow([company, home, about, posts, jobs, people, insights])

print("CSV dataset 'linkedin_companies_dataset.csv' created with realistic synthetic profiles.")
