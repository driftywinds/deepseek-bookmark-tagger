#!/usr/bin/env python3
"""
Bookmark AI Tagger using DeepSeek API (Raindrop.io Support)
Handles nested folder structures from Raindrop.io exports
Usage: python bookmark_tagger.py <bookmarks.html> <api_key> [--dry-run]
"""

import sys
import json
import time
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from urllib.error import HTTPError

class BookmarkItem:
    """Base class for bookmark items"""
    pass

class Bookmark(BookmarkItem):
    def __init__(self, url, title, attrs_dict):
        self.url = url
        self.title = title
        self.attrs = attrs_dict
        existing_tags = attrs_dict.get('tags', '')
        self.tags = [t.strip() for t in existing_tags.split(',') if t.strip()] if existing_tags else []
        self.has_existing_tags = len(self.tags) >= 3

class Folder(BookmarkItem):
    def __init__(self, name, attrs_dict):
        self.name = name
        self.attrs = attrs_dict
        self.items = []

class BookmarkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.root_items = []
        self.folder_stack = []
        self.current_link = None
        self.current_folder = None
        self.pending_folder_name = None
        self.in_h3 = False
        self.in_description = False
        self.current_description = ""
    
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'h3':
            self.in_h3 = True
            self.pending_folder_name = ""
            self.current_folder = Folder("", attrs_dict)
        
        elif tag == 'dl':
            # Starting a new level - if we have a pending folder, push it
            if self.current_folder is not None:
                if self.folder_stack:
                    self.folder_stack[-1].items.append(self.current_folder)
                else:
                    self.root_items.append(self.current_folder)
                self.folder_stack.append(self.current_folder)
                self.current_folder = None
        
        elif tag == 'a':
            href = attrs_dict.get('href', '')
            if href.startswith('http'):
                self.current_link = Bookmark(href, '', attrs_dict)
        
        elif tag == 'dd':
            self.in_description = True
            self.current_description = ""
    
    def handle_data(self, data):
        if self.in_h3:
            self.pending_folder_name += data.strip()
        elif self.current_link is not None:
            self.current_link.title += data.strip()
        elif self.in_description:
            self.current_description += data.strip()
    
    def handle_endtag(self, tag):
        if tag == 'h3':
            self.in_h3 = False
            if self.current_folder:
                self.current_folder.name = self.pending_folder_name
        
        elif tag == 'dl':
            # Closing a level
            if self.folder_stack:
                self.folder_stack.pop()
        
        elif tag == 'a' and self.current_link is not None:
            if self.current_link.title:
                if self.folder_stack:
                    self.folder_stack[-1].items.append(self.current_link)
                else:
                    self.root_items.append(self.current_link)
            self.current_link = None
        
        elif tag == 'dd':
            self.in_description = False
            self.current_description = ""

def collect_all_bookmarks(items):
    """Recursively collect all bookmarks from nested structure"""
    bookmarks = []
    for item in items:
        if isinstance(item, Bookmark):
            bookmarks.append(item)
        elif isinstance(item, Folder):
            bookmarks.extend(collect_all_bookmarks(item.items))
    return bookmarks

def generate_tags(bookmark, api_key, dry_run=False):
    """Call DeepSeek API to generate tags for a bookmark"""
    if dry_run:
        # Return mock tags for dry run
        return ["tag1", "tag2", "tag3"]
    
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
                "content": f"Generate tags for this bookmark:\nTitle: {bookmark.title}\nURL: {bookmark.url}"
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

def build_html_from_items(items, indent=1):
    """Recursively build HTML from nested structure"""
    html = ""
    tab = "    " * indent
    
    for item in items:
        if isinstance(item, Folder):
            # Build folder attributes
            folder_attrs = ' '.join([f'{k.upper()}="{v}"' for k, v in item.attrs.items()])
            html += f'{tab}<DT><H3 {folder_attrs}>{item.name}</H3>\n'
            html += f'{tab}<DL><p>\n'
            html += build_html_from_items(item.items, indent + 1)
            html += f'{tab}</DL><p>\n'
        
        elif isinstance(item, Bookmark):
            # Build bookmark attributes
            attrs_copy = item.attrs.copy()
            tags_str = ', '.join(item.tags)
            attrs_copy['tags'] = tags_str
            
            bookmark_attrs = ' '.join([f'{k.upper()}="{v}"' for k, v in attrs_copy.items()])
            html += f'{tab}<DT><A HREF="{item.url}" {bookmark_attrs}>{item.title}</A>\n'
            if tags_str:
                html += f'{tab}<DD>Tags: {tags_str}\n'
    
    return html

def create_tagged_html(root_items, output_file):
    """Create a new HTML file with tagged bookmarks maintaining folder structure"""
    html = """<!DOCTYPE NETSCAPE-Bookmark-file-1>
<!-- This is an automatically generated file. -->
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DL><p>
"""
    
    html += build_html_from_items(root_items, indent=1)
    html += "</DL><p>"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

def print_structure(items, indent=0):
    """Print the bookmark structure for dry run"""
    prefix = "  " * indent
    for item in items:
        if isinstance(item, Folder):
            print(f"{prefix}📁 {item.name} ({len(item.items)} items)")
            print_structure(item.items, indent + 1)
        elif isinstance(item, Bookmark):
            tag_status = f"[{len(item.tags)} tags]" if item.tags else "[no tags]"
            print(f"{prefix}🔖 {item.title[:50]}... {tag_status}")

def main():
    # Parse arguments
    dry_run = False
    args = sys.argv[1:]
    
    if '--dry-run' in args:
        dry_run = True
        args.remove('--dry-run')
    
    if len(args) < 1 or len(args) > 2:
        print("Usage: python bookmark_tagger.py <bookmarks.html> [<api_key>] [--dry-run]")
        print("\nExamples:")
        print("  python bookmark_tagger.py bookmarks.html --dry-run")
        print("  python bookmark_tagger.py bookmarks.html sk-xxxxxxxxxxxxx")
        print("  python bookmark_tagger.py bookmarks.html sk-xxxxxxxxxxxxx --dry-run")
        print("\nOptions:")
        print("  --dry-run    Preview structure and simulate tagging without using API quota")
        sys.exit(1)
    
    input_file = args[0]
    api_key = args[1] if len(args) > 1 else "dummy-key-for-dry-run"
    
    if not dry_run and (not api_key or api_key == "dummy-key-for-dry-run"):
        print("❌ Error: API key required when not using --dry-run mode")
        sys.exit(1)
    
    mode_label = "DRY RUN MODE - No API calls will be made" if dry_run else "Live Mode"
    print(f"🏷️  Bookmark AI Tagger (Raindrop.io Edition)")
    print("=" * 50)
    print(f"Mode: {mode_label}")
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
    
    # Collect all bookmarks from nested structure
    all_bookmarks = collect_all_bookmarks(parser.root_items)
    
    print(f"✓ Found {len(all_bookmarks)} bookmarks in nested folder structure")
    
    if len(all_bookmarks) == 0:
        print("❌ No bookmarks found in the file")
        sys.exit(1)
    
    # Show structure in dry run mode
    if dry_run:
        print("\n📂 Bookmark Structure:")
        print("-" * 50)
        print_structure(parser.root_items)
        print("-" * 50)
    
    # Process bookmarks
    action_word = "Simulating" if dry_run else "Processing"
    print(f"\n🤖 {action_word} bookmarks with DeepSeek AI...")
    print("-" * 50)
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for i, bookmark in enumerate(all_bookmarks, 1):
        print(f"\n[{i}/{len(all_bookmarks)}] {bookmark.title[:60]}...")
        
        # Skip bookmarks that already have 3+ tags
        if bookmark.has_existing_tags:
            skipped_count += 1
            print(f"  ⊘ Skipped (already has {len(bookmark.tags)} tags: {', '.join(bookmark.tags)})")
            continue
        
        try:
            tags = generate_tags(bookmark, api_key, dry_run=dry_run)
            bookmark.tags = tags
            success_count += 1
            tag_label = "(simulated)" if dry_run else ""
            print(f"  ✓ Tags {tag_label}: {', '.join(tags)}")
            
            # Rate limiting - wait 0.5 seconds between requests (skip in dry run)
            if i < len(all_bookmarks) and not dry_run:
                time.sleep(0.5)
                
        except Exception as e:
            error_count += 1
            bookmark.tags = []
            print(f"  ✗ Error: {str(e)}")
    
    # Save results
    output_file = "bookmarks_tagged_preview.html" if dry_run else "bookmarks_tagged.html"
    save_label = "preview" if dry_run else "final"
    print(f"\n💾 Saving {save_label} tagged bookmarks to: {output_file}")
    create_tagged_html(parser.root_items, output_file)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"  Total bookmarks: {len(all_bookmarks)}")
    print(f"  Skipped (already tagged): {skipped_count}")
    print(f"  Successfully tagged: {success_count}")
    print(f"  Errors: {error_count}")
    
    if dry_run:
        print(f"\n✅ Dry run complete! Preview saved to '{output_file}'")
        print("   No API quota was used.")
        print("   To run for real, remove the --dry-run flag.")
    else:
        print(f"\n✅ Done! Import '{output_file}' into your browser.")
        print("   Folder structure has been preserved.")

if __name__ == "__main__":
    main()
