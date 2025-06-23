import pandas as pd
import re
from urllib.parse import urlparse
from tld import get_tld


def process_single_url(url):
    # 1. URL Length
    url_len = len(url)

    # 2. Process domain (TLD extraction)
    def process_tld(url):
        try:
            res = get_tld(url, as_object=True, fail_silently=False, fix_protocol=True)
            return res.parsed_url.netloc
        except:
            return None

    domain = process_tld(url)

    # 3. Count the number of specific characters in URL
    features = ['@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', '"', ',', '//']
    feature_counts = {feature: url.count(feature) for feature in features}

    # 4. Check for abnormal URL pattern (repeating hostname)
    def abnormal_url(url):
        hostname = urlparse(url).hostname
        return 1 if re.search(hostname, url) else 0

    abnormal_url_flag = abnormal_url(url)

    # 5. Check if the URL is using HTTPS
    def httpSecure(url):
        return 1 if urlparse(url).scheme == 'https' else 0

    https_flag = httpSecure(url)

    # 6. Count digits in the URL
    def digit_count(url):
        return sum(1 for char in url if char.isnumeric())

    digit_count_value = digit_count(url)

    # 7. Count letters in the URL
    def letter_count(url):
        return sum(1 for char in url if char.isalpha())

    letter_count_value = letter_count(url)

    # 8. Check if URL is from a shortening service
    def shortening_service(url):
        match = re.search(r'bit\.ly|goo\.gl|t\.co|tinyurl|adf\.ly|url4\.eu|short\.to|qr\.net|1url\.com', url)
        return 1 if match else 0

    shortening_flag = shortening_service(url)

    # 9. Count the number of directories in the URL path
    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')

    dir_count = no_of_dir(url)

    # 10. Check for suspicious words in URL (e.g., 'login', 'paypal')
    def suspicious_words(url):
        match = re.search(r'PayPal|login|signin|bank|account|update|free|service|bonus|ebayisapi|webscr', url)
        return 1 if match else 0

    suspicious_flag = suspicious_words(url)

    # 11. Calculate hostname length
    hostname_length = len(urlparse(url).netloc)

    # 12. Count the number of uppercase letters in the URL
    upper_count = sum(1 for char in url if char.isupper())

    # 13. Count the number of lowercase letters in the URL
    lower_count = sum(1 for char in url if char.islower())

    # 14. Check if the URL has a "www" prefix
    has_www = 1 if 'www.' in url else 0

    # 15. Count number of subdomains (split by '.')
    subdomain_count = len(urlparse(url).hostname.split('.')) - 2 if urlparse(url).hostname else 0

    # 16. Count the number of query parameters
    query_count = len(urlparse(url).query.split('&')) if urlparse(url).query else 0

    # 17. Count the number of fragments in the URL
    fragment_count = 1 if urlparse(url).fragment else 0

    # 18. Check if the URL uses a port number
    has_port = 1 if urlparse(url).port else 0

    # 19. Count the number of slashes in the URL
    slash_count = url.count('/')

    # 20. Check if the URL uses a path
    has_path = 1 if urlparse(url).path else 0

    # 21. Check if the URL contains "http"
    contains_http = 1 if 'http' in url else 0

    # 22. Check if the URL contains a valid top-level domain
    valid_tld = 1 if process_tld(url) else 0

    # 23. Check if the URL contains a valid domain (e.g., example.com)
    has_valid_domain = 1 if domain else 0

    # 24. Check if the URL contains the string "secure"
    contains_secure = 1 if 'secure' in url else 0

    # 25. Check if the URL contains the string "login"
    contains_login = 1 if 'login' in url else 0

    # 26. Check if the URL contains the string "signup"
    contains_signup = 1 if 'signup' in url else 0

    # Combine all features into a dictionary
    features_dict = {
        'url_len': url_len,
        '@': feature_counts['@'],
        '?': feature_counts['?'],
        '-': feature_counts['-'],
        '=': feature_counts['='],
        '.': feature_counts['.'],
        '#': feature_counts['#'],
        '%': feature_counts['%'],
        '+': feature_counts['+'],
        '$': feature_counts['$'],
        '!': feature_counts['!'],
        '*': feature_counts['*'],
        ',': feature_counts[','],
        '//': feature_counts['//'],
        'abnormal_url': abnormal_url_flag,
        'https': https_flag,
        'digits': digit_count_value,
        'letters': letter_count_value,
        'Shortening_Service': shortening_flag,
        'count_dir': dir_count,
        'sus_url': suspicious_flag,
        'hostname_length': hostname_length
    }

    # Convert to a DataFrame (for easier handling and saving)
    df_single = pd.DataFrame([features_dict])

    #df_single['Category'] = -1  # Here, we set -1 because it's unknown for a single URL
    # Save to CSV
    df_single.to_csv('url_test.csv', index=False)

    return df_single


# Example usage:
url = "http://example.com/login?user=test"
df_single = process_single_url(url)
print(df_single)
