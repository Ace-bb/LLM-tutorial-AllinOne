import json, re, requests
from urllib import parse
from tqdm import tqdm

IMAGE_CLOUD_URL="https://gitee.com/Ace_bb/transformer/raw/main"

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

def trans_cell_one_img(img_row):
    img_src = re.search(r'src="(.*?)"', img_row).group(1)
    img_url = f"{IMAGE_CLOUD_URL}/{img_src}"
    return img_row.replace(img_src, img_url)

def trans_cell_multi_img(multi_img_content):
    '''<p align="center">
        <img src="imgs/position/position_encoding_ex.png" width="50%">
    </p>'''
    # 获取图片地址src
    img_rows = multi_img_content
    for rid, row in enumerate(img_rows):
        if row.strip().startswith('<img'):
            img_src = re.search(r'src="(.*?)"', row).group(1)
            img_url = f"{IMAGE_CLOUD_URL}/{img_src}"
            img_rows[rid] = row.replace(img_src, img_url).strip(" ")
    # print('\n'.join(img_rows))
    return '\n'.join(img_rows) + '\n'

def trans_code_cell_2(code_source):
    code_res = ''
    cell_codes = []
    return f"```python\n{''.join(code_source)}\n```"

def read_cells(notebook):
    cells = notebook['cells']
    artical_content =  []
    H1_Title_Id= 0
    for i, cell in tqdm(enumerate(cells), total=len(cells)):
        if cell['cell_type'] == 'markdown':
            for rid, row in enumerate(cell['source']):
                if row.strip().startswith('<img'):
                    artical_content.append(trans_cell_one_img(row)) # 单张图片
                elif row.strip().startswith('<p'):
                    multi_img_content = []
                    for j in range(rid, len(cell['source'])):
                        if cell['source'][j].strip() == '</p>':
                            multi_img_content = cell['source'][rid:j+1]
                            cell['source'][rid:j+1] = ['']*(j-rid+1)
                            break
                    artical_content.append(trans_cell_multi_img(multi_img_content)) # 多张图片
                else:
                    artical_content.append(row)
        elif cell['cell_type'] == 'code':
            artical_content.append(trans_code_cell_2(cell['source']))
    
    return artical_content

def parse_notebook(note_book_path):
    notebook = read_notebook(note_book_path)
    artical_content = read_cells(notebook)
    for _id, item in enumerate(artical_content):
        if item==None:
            print(_id)
            artical_content[_id] = '\n'
            
    with open(note_book_path.replace('.ipynb', '.md'), 'w', encoding='utf-8') as f:
        f.write(''.join(artical_content))
        
if __name__ == '__main__':
    parse_notebook('2-主流模型架构/2.2-LLaMA/LLaMA.ipynb')
    # access_token = get_weixin_access_token()['access_token'] #"85_5jStf6LK3xA73EyKauZR72t4fjpcD_Bkv96N2Y3u8Xz0iN7UnEA7DY0y9oBBlF5y3lCuVoLvgSr_WUKuht_tiDHZV3vBK9TtGmf2kneCcw6sRp-as9GzEDBJ1PEZTLaAJATNF" # get_weixin_access_token()['access_token']
    # print(upload_image_to_weixin('image.png', access_token))