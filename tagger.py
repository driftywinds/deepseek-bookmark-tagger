#!/usr/bin/env python3
"""
Bookmark AI Tagger using DeepSeek API
Usage: python bookmark_tagger.py <bookmarks.html> <api_key>
"""

import sys
import json
import time
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from urllib.error import HTTPError

class BookmarkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.bookmarks = []
        self.current_link = None
    
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            attrs_dict = dict(attrs)
            href = attrs_dict.get('href', '')
            if href.startswith('http'):
                # Check if the link already has tags
                existing_tags = attrs_dict.get('tags', '')
                tags_list = [t.strip() for t in existing_tags.split(',') if t.strip()] if existing_tags else []
                self.current_link = {
                    'url': href, 
                    'title': '',
                    'tags': tags_list,
                    'has_existing_tags': len(tags_list) >= 3
                }
    
    def handle_data(self, data):
        if self.current_link is not None:
            self.current_link['title'] += data.strip()
    
    def handle_endtag(self, tag):
        if tag == 'a' and self.current_link is not None:
            if self.current_link['title']:
                self.bookmarks.append(self.current_link)
            self.current_link = None

def generate_tags(bookmark, api_key):
    """Call DeepSeek API to generate tags for a bookmark"""
    url = "https://api.deepseek.com/v1/chat/completions"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a bookmark tagging assistant. Generate 3-5 relevant, concise tags for the given bookmark. Return ONLY the tags as a comma-separated list, nothing else."
            },
            {
                "role": "user",
                "content": f"Generate tags for this bookmark:\nTitle: {bookmark['title']}\nURL: {bookmark['url']}"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    req = Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
    
    try:
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            tags_text = data['choices'][0]['message']['content'].strip()
            tags = [t.strip() for t in tags_text.split(',') if t.strip()]
            return tags
    except HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise Exception(f"API Error: {e.code} - {error_body}")

def create_tagged_html(bookmarks, output_file):
    """Create a new HTML file with tagged bookmarks"""
    html = """<!DOCTYPE NETSCAPE-Bookmark-file-1>
<!-- This is an automatically generated file. -->
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DL><p>
"""
    
    for bookmark in bookmarks:
        tags = ', '.join(bookmark.get('tags', []))
        html += f'    <DT><A HREF="{bookmark["url"]}" TAGS="{tags}">{bookmark["title"]}</A>\n'
        if tags:
            html += f'    <DD>Tags: {tags}\n'
    
    html += "</DL><p>"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

def main():
    if len(sys.argv) != 3:
        print("Usage: python bookmark_tagger.py <bookmarks.html> <api_key>")
        print("\nExample:")
        print("  python bookmark_tagger.py bookmarks.html sk-xxxxxxxxxxxxx")
        sys.exit(1)
    
    input_file = sys.argv[1]
    api_key = sys.argv[2]
    
    print("🏷️  Bookmark AI Tagger")
    print("=" * 50)
    
    # Parse bookmarks
    print(f"\n📖 Reading bookmarks from: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found")
        sys.exit(1)
    
    parser = BookmarkParser()
    parser.feed(html_content)
    bookmarks = parser.bookmarks
    
    print(f"✓ Found {len(bookmarks)} bookmarks")
    
    if len(bookmarks) == 0:
        print("❌ No bookmarks found in the file")
        sys.exit(1)
    
    # Process bookmarks
    print(f"\n🤖 Processing bookmarks with DeepSeek AI...")
    print("-" * 50)
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for i, bookmark in enumerate(bookmarks, 1):
        print(f"\n[{i}/{len(bookmarks)}] {bookmark['title'][:60]}...")
        
        # Skip bookmarks that already have 3+ tags
        if bookmark.get('has_existing_tags', False):
            skipped_count += 1
            print(f"  ⊘ Skipped (already has {len(bookmark['tags'])} tags: {', '.join(bookmark['tags'])})")
            continue
        
        try:
            tags = generate_tags(bookmark, api_key)
            bookmark['tags'] = tags
            success_count += 1
            print(f"  ✓ Tags: {', '.join(tags)}")
            
            # Rate limiting - wait 0.5 seconds between requests
            if i < len(bookmarks):
                time.sleep(0.5)
                
        except Exception as e:
            error_count += 1
            bookmark['tags'] = []
            print(f"  ✗ Error: {str(e)}")
    
    # Save results
    output_file = "bookmarks_tagged.html"
    print(f"\n💾 Saving tagged bookmarks to: {output_file}")
    create_tagged_html(bookmarks, output_file)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"  Total bookmarks: {len(bookmarks)}")
    print(f"  Skipped (already tagged): {skipped_count}")
    print(f"  Successfully tagged: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"\n✅ Done! Import '{output_file}' into your browser.")

if __name__ == "__main__":
    main()
