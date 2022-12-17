from typing import List

import IPython.display as ipd
import pandas as pd

class JupyterNotebookUtil():
    def __init__(self) -> None:
        self.lower_is_better_symbol:str = "↓"
        self.higher_is_better_symbol:str = "↑"

    def get_html_code_from_srcdir(self,type:str,dir:str) -> str:
        if type == "audio":
            return f"""<audio controls><source src="{dir}" type="audio/wav"></audio></td>"""
        elif type == "img":
            return f"""<img src="{dir}">"""
    
    def display_html_list(self,html_list:list) -> None:
        for html_result in html_list:
            ipd.display(ipd.HTML(html_result))
    
    def pandas_list_to_html(self,pandas_list: List[dict]) -> str:
        df = pd.DataFrame(pandas_list)
        return df.to_html(escape=False,index=False)
    