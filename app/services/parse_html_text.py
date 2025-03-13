import re
from bs4 import BeautifulSoup

class HTMLParser:
    def __init__(self, html_text: str, with_blocks: bool = False):
        self.html_text = self.clean_system_prompt(html_text)
        self.with_blocks = with_blocks
        self.soup = BeautifulSoup(self.html_text, 'lxml')
        self.body = self.soup.find('body')
    
    def clean_system_prompt(self, raw_text: str) -> str:
        match = re.search(r"```html\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            return match.group(1)
        return raw_text

    def parse(self):
        if not self.body:
            return {"text": ""}
        if self.with_blocks:
            return self._extract_with_blocks()
        else:
            return self._extract_plain_text()

    def _extract_with_blocks(self):
        blocks = []
        for element in self.body.find_all(True, {"data-bbox": True}):
            block = element.get("data-bbox", "")
            text = element.get_text(separator=" ", strip=True)
            if text:
                blocks.append({
                    "block": block,
                    "text": text
                })
        return {"blocks": blocks}

    def _extract_plain_text(self):
        texts = []
        for element in self.body.find_all(recursive=True):
            if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'li']:
                text = element.get_text(separator=" ", strip=True)
                if text:
                    texts.append(text)
        combined_text = " ".join(texts)
        cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()
        return {"text": cleaned_text}

        return raw_text

    def parse(self):
        if not self.body:
            return {"text": ""}
        if self.with_blocks:
            return self._extract_with_blocks()
        else:
            return self._extract_plain_text()

    def _extract_with_blocks(self):
        blocks = []
        for element in self.body.find_all(True, {"data-bbox": True}):
            block = element.get("data-bbox", "")
            text = element.get_text(separator=" ", strip=True)
            if text:
                blocks.append({
                    "block": block,
                    "text": text
                })
        return {"blocks": blocks}

    def _extract_plain_text(self):
        texts = []
        for element in self.body.find_all(recursive=True):
            if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'li']:
                text = element.get_text(separator=" ", strip=True)
                if text:
                    texts.append(text)
        combined_text = " ".join(texts)
        cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()
        return {"text": cleaned_text}
