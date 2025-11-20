from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

import time, random
import pandas as pd

url = 'xxx'

def human_pause(a=2.0, b=4.0):
    time.sleep(random.uniform(a, b))
    

def go_to_page(page_no: int):
    
    
    xpath = f"//div[starts-with(@class,'hermes-PaginationBar-module')]//span[normalize-space(text())='{page_no}']"

    elem = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", elem)
    human_pause()
    driver.execute_script("arguments[0].click();", elem)
    human_pause(1.5, 3.0)   



def safe_process_page(page, process_function):
    try:
        return process_function()
    except TimeoutException as e:
        print(f"[{page}. sayfa] TimeoutException — 30-60 sn bekleniyor:", e)
        time.sleep(random.uniform(30, 60))
        return None
    except StaleElementReferenceException as e:
        print(f"[{page}. sayfa] StaleElement — 20-40 sn bekleniyor:", e)
        time.sleep(random.uniform(20, 40))
        return None
    except Exception as e:
        print(f"[{page}. sayfa] Beklenmeyen hata — 40-70 sn bekleniyor:", e)
        time.sleep(random.uniform(40, 70))
        return None



options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')   #tam ekran
options.add_experimental_option('detach', True)     #kapanmamasını sagla
# =============================================================================
# options.add_argument('--headless')
# =============================================================================



driver = webdriver.Chrome(options=options)
reviews_url = url + "-yorumlari"
driver.get(reviews_url)
human_pause()
wait = WebDriverWait(driver, 15)


try:
    cookie_btn = wait.until(
        EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(., 'Kabul') or contains(., 'kabul')]")
        )
    )
    cookie_btn.click()
    human_pause()
except:
    pass  





pager_spans = driver.find_elements(
    By.XPATH,
    "//div[starts-with(@class,'hermes-PaginationBar-module')]//span[normalize-space()]"
)

page_nums = []
for s in pager_spans:
    t = s.text.strip()
    if t.isdigit():        # sadece sayılara bak
        page_nums.append(int(t))

max_page = max(page_nums)   # örneğin 100
print("Toplam sayfa:", max_page)


wait.until(
    EC.presence_of_all_elements_located(
        (By.CSS_SELECTOR, "div[class^='hermes-ReviewCard-module']")
    )
)


# =============================================================================
# review_cards = driver.find_elements(By.CSS_SELECTOR, "div[class^='hermes-ReviewCard-module']")
# =============================================================================




    
comments = []
seen = set()

aylar = ["Ocak", "Şubat", "Mart", "Nisan",
          "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]



for page in range(1, max_page + 1):
    print(f"== {page}. sayfa ==")

    if page > 1:
        go_to_page(page)     # 1. sayfa zaten açık
        
    if page % 10 == 0:
        print("Küçük mola...")
        time.sleep(random.uniform(15, 30))
        
# =============================================================================
#     
#     if "captcha" in driver.page_source.lower():
#         print("⚠ CAPTCHA ALGILANDI! Scraping durduruldu.")
#         break  
# =============================================================================
        
        
    def process():
        # Yorum kartlarını çek
        review_cards = driver.find_elements(By.CSS_SELECTOR, "div[class^='hermes-ReviewCard-module']")
        
        
        added_count = 0
        
        

        
        
        for card in review_cards:
            # 1) Yorum
            try:
                comment_span = card.find_element( By.XPATH, ".//span[contains(@style, 'text-align') and normalize-space()]")
                text = comment_span.text.strip()
            except:
                continue
            
            if "Değerlendirilen özellikler" in text:
                continue

            if not text or text in seen:
                continue

            seen.add(text)

            # 2) Yıldız
            star_count = None
            try:
                rating_div = card.find_element(
                    By.CSS_SELECTOR, "div[class^='hermes-RatingPointer-module']"
                )
                stars = rating_div.find_elements(By.CSS_SELECTOR, "div.star")
                star_count = len(stars)
            except:
                star_count = None

            # 3) Tarih 
            date_text = None
            try:
                spans = card.find_elements(By.TAG_NAME, "span")
                for s in spans:
                    t = s.text.strip()
                    if any(ay in t for ay in aylar):
                        date_text = t
                        break
            except:
                date_text = None

            comments.append({
                "comment": text,
                "stars": star_count,
                "date": date_text
            })
            added_count += 1
            
        
        print(f"{page}. sayfadan eklenen yorun: {added_count}")
        
        
        
        
        
        return True  # işlem başarılı
    
    safe_process_page(page, process)





# Kontrol
print("Toplam uniq yorum:", len(comments))


df = pd.DataFrame(comments)
df.to_csv("reviews.csv", index=False, encoding='utf-8-sig')






