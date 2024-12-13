import json, re, requests
from urllib import parse
from tqdm import tqdm
import shutil, os

IMAGE_CLOUD_URL="https://gitee.com/Ace_bb/static_resource_cloud/raw/master"

QA_ID = 0
# 发送Http给Get请求
def send_http_get(url):
    import requests
    response = requests.get(url)
    return response

def read_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    return notebook

def upload_image_to_weixin(image_path, access_token):
    url = f"https://api.weixin.qq.com/cgi-bin/media/uploadimg?access_token={access_token}"
    files = {'media': open(image_path, 'rb')}
    response = requests.post(url, files=files)
    return response.json()

def trans_cell_one_img(img_row, folder_path):
    img_src = re.search(r'src="(.*?)"', img_row).group(1)
    folder_path,img_src = folder_path.replace('./', ''), img_src.replace('./', '')
    if not os.path.exists(f"repo/static_resource_cloud/{folder_path}/_img"): os.makedirs(f"repo/static_resource_cloud/{folder_path}/_img")
    shutil.copy(f"{folder_path}/{img_src}", f"repo/static_resource_cloud/{folder_path}/{img_src}")
    img_src = f"{folder_path}/{img_src}"
    img_url = f"{IMAGE_CLOUD_URL}/{img_src}"
    return img_row.replace(img_src, img_url)

def trans_cell_multi_img(multi_img_content, folder_path):
    '''<p align="center">
        <img src="imgs/position/position_encoding_ex.png" width="50%">
    </p>'''
    # 获取图片地址src
    img_rows = multi_img_content
    for rid, row in enumerate(img_rows):
        if row.strip().startswith('<img'):
            ori_img_src = re.search(r'src="(.*?)"', row).group(1)
            folder_path,img_src = folder_path.replace('./', ''), ori_img_src.replace('./', '')
            if not os.path.exists(f"repo/static_resource_cloud/{folder_path}/_img"): os.makedirs(f"repo/static_resource_cloud/{folder_path}/_img")
            shutil.copy(f"{folder_path}/{img_src}", f"repo/static_resource_cloud/{folder_path}/{img_src}")
            img_src = f"{folder_path}/{img_src}"
            img_url = f"{IMAGE_CLOUD_URL}/{img_src}"
            img_rows[rid] = row.replace(ori_img_src, img_url).strip(" ")
    # print('\n'.join(img_rows))
    return '\n'.join(img_rows) + '\n'

def trans_code_cell_2(code_source):
    code_res = ''
    cell_codes = []
    return f"```python\n{''.join(code_source)}\n```"

def parse_notebook_content(folder_path, notebook_name):
    notebook_path = f"{folder_path}/{notebook_name}"
    notebook = read_notebook(notebook_path)
    cells = notebook['cells']
    artical_content =  []
    H1_Title_Id= 0
    for i, cell in tqdm(enumerate(cells), total=len(cells), desc=notebook_path):
        if cell['cell_type'] == 'markdown':
            for rid, row in enumerate(cell['source']):
                if row.strip().startswith('<img'):
                    one_img_Content = trans_cell_one_img(row, folder_path)
                    artical_content.append(one_img_Content) # 单张图片
                elif row.strip().startswith('<p'):
                    multi_img_content = []
                    for j in range(rid, len(cell['source'])):
                        if cell['source'][j].strip() == '</p>':
                            multi_img_content = cell['source'][rid:j+1]
                            cell['source'][rid:j+1] = ['']*(j-rid+1)
                            break
                    mic = trans_cell_multi_img(multi_img_content, folder_path)
                    artical_content.append(mic) # 多张图片
                else:
                    artical_content.append(row)
        elif cell['cell_type'] == 'code':
            artical_content.append("\n\n" + trans_code_cell_2(cell['source']) + "\n\n")
    
    return artical_content

def parse_notebook():
    for l_1_folder in os.listdir("./"):
        if l_1_folder==".git": continue
        if not os.path.isdir(f"./{l_1_folder}"): continue
        for l_2_folder in os.listdir(f"./{l_1_folder}"):
            if l_2_folder=="1.2-Transformer": continue
            if not os.path.isdir(f"./{l_1_folder}/{l_2_folder}"): continue
            for f_name in os.listdir(f"./{l_1_folder}/{l_2_folder}"):
                note_book_path = f"./{l_1_folder}/{l_2_folder}/{f_name}"
                # if f_name !="LLaMA.ipynb": continue
                try:
                    if f_name.endswith(".ipynb"): 
                        artical_content = parse_notebook_content(f"./{l_1_folder}/{l_2_folder}", f_name)
                        for _id, item in enumerate(artical_content):
                            if item==None:
                                print(_id)
                                artical_content[_id] = '\n'
                        res_save_path = "repo/parse_notebooks/" + note_book_path.replace('.ipynb', '.md')
                        base_dir = os.path.dirname(res_save_path)
                        if not os.path.exists(base_dir): os.makedirs(base_dir)
                        with open(res_save_path, 'w', encoding='utf-8') as f:
                            f.write(''.join(artical_content))
                    elif f_name.endswith('.md'):
                        ...
                    else:
                        continue
                except:
                    continue

if __name__ == '__main__':
    parse_notebook()
    # access_token = get_weixin_access_token()['access_token'] #"85_5jStf6LK3xA73EyKauZR72t4fjpcD_Bkv96N2Y3u8Xz0iN7UnEA7DY0y9oBBlF5y3lCuVoLvgSr_WUKuht_tiDHZV3vBK9TtGmf2kneCcw6sRp-as9GzEDBJ1PEZTLaAJATNF" # get_weixin_access_token()['access_token']
    # print(upload_image_to_weixin('image.png', access_token))