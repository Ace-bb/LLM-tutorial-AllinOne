from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json, os
from tqdm import tqdm
import numpy as np
from conf.Tools import Tools

def get_static_url_content_after_click(driver, url, button_text):
    """
    使用 Selenium WebDriver 模拟点击操作后获取静态网页内容。

    Parameters:
    - driver: WebDriver对象，用于控制浏览器
    - url: 要打开的网页URL
    - button_text: 页面中的按钮文本

    Returns:
    - bsObj: BeautifulSoup对象，包含解析后的页面内容
    """
    # 打开网页
    driver.get(url)

    # 使用 WebDriverWait 等待页面元素加载
    button = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH,
                                          f"//div[@class='tag-button' and text()='{button_text}'] | //div[@class='tag-button active' and text()='{button_text}']"))
    )

    # 点击按钮
    button.click()

    # 获取动态加载后的内容
    page_source = driver.page_source

    # 使用BeautifulSoup解析页面
    bsObj = BeautifulSoup(page_source, 'lxml')

    return bsObj


def start_crawler(subject):
    """
    开始爬取特定科室的链接

    Parameters:
    - subject: 科室的编号或名称

    Returns:
    - subject_link: 包含科室链接列表的列表
    """
    # 设置 Chrome 驱动的路径
    chrome_path = "C:\\Program Files\\Google\\Chrome\\Application\\chromedriver.exe"

    # 创建 Chrome 驱动
    chrome_service = ChromeService(chrome_path)
    driver = webdriver.Chrome(service=chrome_service)

    url_template = 'https://dxy.com/diseases/%s'

    # 创建一个列表用于存储每个科室的链接列表
    subject_link = []

    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        url = url_template % (subject)
        try:
            # 尝试获取当前字母的页面
            bsObj = get_static_url_content_after_click(driver, url, letter)

            # 找到所有class为"section-card common-card-link"的a标签
            section_cards = bsObj.find_all('a', class_='section-card common-card-link')

            # 提取每个a标签中的href属性
            href_list = [card['href'] for card in section_cards]

            # print(f'Subject: {subject}, Letter: {letter}')
            # print(href_list)

            # 将链接添加到列表中
            subject_link.append(href_list)

        except TimeoutException:
            # 如果超时异常发生，说明当前字母不存在，跳过本轮循环
            print(f"Subject: {subject}, Letter: {letter} not found, skipping...")
            continue

    # 关闭浏览器
    driver.quit()

    return subject_link


def get_static_url_content(url):
    """
    使用 Selenium WebDriver 获取指定 URL 的静态网页内容。

    Parameters:
    - url (str): 要获取内容的网页链接。

    Returns:
    - bsObj: BeautifulSoup对象，用于解析HTML。
    """
    # 设置 Chrome 驱动的路径
    chrome_path = "C:\\Program Files\\Google\\Chrome\\Application\\chromedriver.exe"

    # 创建 Chrome 驱动
    chrome_service = ChromeService(chrome_path)
    driver = webdriver.Chrome(service=chrome_service)

    try:
        # 打开指定的 URL
        driver.get(url)

        # 获取网页内容
        page_content = driver.page_source

        # 使用BeautifulSoup解析HTML
        bsObj = BeautifulSoup(page_content, 'lxml')

    except Exception as e:
        print(f"获取网页内容时发生错误：{e}")
        bsObj = None

    # 关闭浏览器
    driver.quit()
    return bsObj


def start_get_disease_info(url, _id):
    try:
        res = get_disease_info(url)
        return True, {"id": _id, "data": res}
    except:
        return False, {"url": url}
    
def get_disease_info(url):
    """
    获取特定疾病的详细信息

    Parameters:
    - link: 疾病页面的链接

    Returns:
    - disease_info: 包含疾病信息的字典或其他数据结构
    """
    
    bsObj = get_static_url_content(url)

    # 返回一个包含疾病信息的字典
    disease_info = {
        "url": url,
        'name': None,
        'introduction': [],
        'symptoms': [],
        'causes': [],
        'diagnosis': [],
        'treatments': [],
        'lifestyle': [],
        'prevention': [],
    }

    # 提取疾病名称
    name_element = bsObj.find('div', class_='high-light tag-content-title')
    if name_element:
        disease_info['name'] = name_element.text.strip()

    # 提取疾病信息
    disease_details = list()
    disease_detail_card = bsObj.find_all('div', class_="disease-detail-card")
    for disease_card in disease_detail_card:
        card_title = disease_card.find('p', 'disease-detail-card-title')
        disease_content = {}
        disease_content["title"] = card_title.get_text(strip=True)
        disease_content["content"] = list()
        disease_content["html"] = str(disease_card)
        child_content = {"Question": "", "Answer": ""}
        for child in disease_card.find("div", "html-parse tag-html").children:
            if child.name in ["h1", "h2", "h3"] :
                if child_content["Question"] != "" or len(child_content["Question"])!=0: 
                    disease_content["content"].append(child_content)
                child_content={"Question": child.get_text(strip=True), "Answer": ""}
            else:
                child_content["Answer"] += "\n" + child.get_text(strip=True)
        disease_content["content"].append(child_content)
        disease_details.append(disease_content)
    
    disease_info["disease_details"] = disease_details
    disease_info["disease_detail_card_html"] = str(disease_detail_card)
    info_elements = bsObj.find_all('div', class_='html-parse tag-html')
    i = 0
    for info_element in info_elements:
        info_text = info_element.get_text(strip=True)

        # 根据i值确定存储字段
        if i == 0:
            disease_info['introduction'] = info_text.strip()
        elif i == 1:
            disease_info['symptoms'] = info_text.strip()
        elif i == 2:
            disease_info['causes'] = info_text.strip()
        elif i == 3:
            disease_info['diagnosis'] = info_text.strip()
        elif i == 4:
            disease_info['treatments'] = info_text.strip()
        elif i == 5:
            disease_info['lifestyle'] = info_text.strip()
        elif i == 6:
            disease_info['prevention'] = info_text.strip()
        else:
            # 处理未知情况
            pass

        i = i+1
    return disease_info


def write_to_file(file_name, content):
    """
    将内容写入指定的文件（追加模式）。

    Parameters:
    - file_name (str): 要写入的文件的名称。
    - content (dict): 要写入文件的内容，这里假设是一个字典。

    Returns:
    - 无
    """
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            # json_content = json.dumps(content, ensure_ascii=False, indent=2)
            json.dump(content, file, ensure_ascii=False)
            # file.write(json_content + '\n@@@@@@\n')
            print(f"成功写入文件 {file_name}")
    except Exception as e:
        print(f"写入文件 {file_name} 时发生错误：{e}")


def main():
    tools  = Tools()
    # 创建字典用于映射科室数字和名称
    subject_mapping = {'6133': '心血管内科', "6984": "儿科", "2781":"妇产科", "25431": "生殖 遗传"}
    with open("mao.json", 'r', encoding="utf-8") as f:
        subject_mapping = json.load(f)
    # 创建字典用于存储每个科室的疾病链接
    all_subject_links = {}

    # for subject_number, subject_name in subject_mapping.items():
    #     links = start_crawler(subject_number)
    #     with open(f"./subjectLinks/{subject_name}.json", 'w', encoding="utf-8") as f:
    #         json.dump(links, f, ensure_ascii=False)
    #     all_subject_links[subject_name] = links
    for subject_file in os.listdir("./subjectLinks"):
        # if subject_file!="皮肤科.json": continue
        if subject_file not in ["儿科.json", "内分泌科.json", "口腔科.json", "呼吸内科.json", "妇产科.json", "影像检验科.json", "心胸外科.json", "心血管内科.json" ]: continue
        # if os.path.exists(f"./data/{subject_file}"): continue
        with open(f"./subjectLinks/{subject_file}", 'r', encoding="utf-8") as f:
            items = list()
            for it in json.load(f):
                items.extend(it)
            all_subject_links[subject_file.replace('.json', '')] = items
            
    # print(all_subject_links)
    for subject_name, subject_links in all_subject_links.items():
        # 创建一个以科室命名的文件
        file_name = f"./data/{subject_name}.json"
        subject_data = list()
        run_paras = list()
        num = 0
        for link in tqdm(subject_links, desc=f"subject_name"):
            # if link!="https://dxy.com/disease/25864/detail": continue
            run_paras.append((link, 0))
            num+=1
            # if num>=10: break
        subject_data = tools.multi_thread_run(16, start_get_disease_info, run_paras, description=f"{subject_name}")
            # disease_info = get_disease_info(link)
            # subject_data.append(disease_info)
            # print("link: ", link)
                # 将疾病信息写入文件
        write_to_file(file_name, subject_data)
        
def main2():
    tools = Tools()
    for subject_file in os.listdir("./data"):
        if subject_file not in ["儿科.json", "内分泌科.json", "口腔科.json", "呼吸内科.json", "妇产科.json", "影像检验科.json", "心胸外科.json", "心血管内科.json" ]: continue
        # if subject_file != "普外科.json": continue
        subject_data = tools.read_json(f"./data/{subject_file}")
        run_paras = list()
        num=0
        for _id, item in enumerate(subject_data):
            if item["name"]==None or len(item["disease_details"]) == 0:
                run_paras.append((item["url"], _id))
                num+=1
        res_data = tools.multi_thread_run(16, start_get_disease_info, run_paras, description=f"{subject_file}")
        for item in res_data:
            subject_data[item["id"]] = item["data"]
        write_to_file(f"./data/{subject_file}", subject_data)
                
                
def parse():
    tools = Tools()
    for subject_file in os.listdir("./data"):
        if subject_file not in ["儿科.json", "内分泌科.json", "口腔科.json", "呼吸内科.json", "妇产科.json", "影像检验科.json", "心胸外科.json", "心血管内科.json" ]: continue
        # if subject_file != "普外科.json": continue
        subject_data = tools.read_json(f"./data/{subject_file}")
        res_subject_data = [item["data"] for item in subject_data]
        tools.write_2_json(res_subject_data, f"./data/{subject_file}")

def clean():
    tools = Tools()
    total_num = 0
    for subject_file in os.listdir("./data"):
        subject_data = tools.read_json(f"./data/{subject_file}")
        total_num+=len(subject_data)
        # res_subject_data = []
        # tools.write_2_json(res_subject_data, f"./data/{subject_file}")

    print(total_num)
if __name__ == '__main__':
    clean()

