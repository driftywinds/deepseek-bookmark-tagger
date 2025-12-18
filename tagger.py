#!/usr/bin/env python3
"""
Raindrop.io AI Tagger using Direct API
Tags bookmarks directly in Raindrop.io using DeepSeek API
Supports nested collections and skips already-tagged bookmarks

Prerequisites:
1. Create a Raindrop.io app at https://app.raindrop.io/settings/integrations
2. Get test token OR complete OAuth flow
3. Get DeepSeek API key from https://platform.deepseek.com

Usage:
  python raindrop_tagger.py <raindrop_token> <deepseek_api_key> [--dry-run] [--collection-id ID]
  
Examples:
  python raindrop_tagger.py rdp_xxxxx sk-xxxxx --dry-run
  python raindrop_tagger.py rdp_xxxxx sk-xxxxx
  python raindrop_tagger.py rdp_xxxxx sk-xxxxx --collection-id 12345
"""

import sys
import json
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from typing import List, Dict, Optional

class RateLimiter:
    """Track and respect Raindrop API rate limits"""
    
    def __init__(self):
        self.limit = 120  # requests per minute
        self.remaining = 120
        self.reset_time = 0
        self.request_times = []
        
    def update_from_headers(self, headers):
        """Update rate limit info from response headers"""
        try:
            if 'X-RateLimit-Limit' in headers:
                self.limit = int(headers['X-RateLimit-Limit'])
            if 'X-RateLimit-Remaining' in headers:
                self.remaining = int(headers['X-RateLimit-Remaining'])
            if 'X-RateLimit-Reset' in headers:
                self.reset_time = int(headers['X-RateLimit-Reset'])
        except (ValueError, KeyError):
            pass
    
    def wait_if_needed(self):
        """Wait if we're approaching rate limit"""
        current_time = time.time()
        
        # Remove requests older than 60 seconds
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # If we've made 100+ requests in the last minute, wait
        if len(self.request_times) >= 100:
            oldest = self.request_times[0]
            wait_time = 60 - (current_time - oldest) + 1
            if wait_time > 0:
                print(f"    ⏳ Rate limit protection: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.request_times = []
        
        # If API says we're low on remaining requests, add extra delay
        elif self.remaining < 10 and self.remaining > 0:
            print(f"    ⏳ Low rate limit ({self.remaining} remaining): adding delay...")
            time.sleep(2)
        
        # Record this request
        self.request_times.append(time.time())


class RaindropAPI:
    """Raindrop.io API client with rate limiting"""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://api.raindrop.io/rest/v1"
        self.rate_limiter = RateLimiter()
        
    def _make_request(self, endpoint: str, method: str = "GET", data: dict = None, retry_count: int = 0) -> dict:
        """Make HTTP request to Raindrop API with rate limiting and retries"""
        
        # Wait if needed to respect rate limits
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        req_data = json.dumps(data).encode('utf-8') if data else None
        req = Request(url, data=req_data, headers=headers, method=method)
        
        try:
            with urlopen(req, timeout=30) as response:
                # Update rate limit info from headers
                self.rate_limiter.update_from_headers(response.headers)
                return json.loads(response.read().decode('utf-8'))
                
        except HTTPError as e:
            # Update rate limit info even on error
            self.rate_limiter.update_from_headers(e.headers)
            
            # Handle 429 Too Many Requests with exponential backoff
            if e.code == 429 and retry_count < 3:
                wait_time = (2 ** retry_count) * 10  # 10s, 20s, 40s
                print(f"    ⚠️  Rate limited! Waiting {wait_time}s before retry {retry_count + 1}/3...")
                time.sleep(wait_time)
                return self._make_request(endpoint, method, data, retry_count + 1)
            
            error_body = e.read().decode('utf-8')
            raise Exception(f"Raindrop API Error: {e.code} - {error_body}")
    
    def get_root_collections(self) -> List[Dict]:
        """Get all root collections"""
        result = self._make_request("collections")
        return result.get("items", [])
    
    def get_child_collections(self) -> List[Dict]:
        """Get all child/nested collections"""
        result = self._make_request("collections/childrens")
        return result.get("items", [])
    
    def get_raindrops(self, collection_id: int, page: int = 0, perpage: int = 50, nested: bool = False) -> Dict:
        """Get raindrops from a collection"""
        nested_param = "&nested=true" if nested else ""
        endpoint = f"raindrops/{collection_id}?perpage={perpage}&page={page}{nested_param}"
        return self._make_request(endpoint)
    
    def update_raindrop(self, raindrop_id: int, tags: List[str]) -> Dict:
        """Update a raindrop's tags"""
        endpoint = f"raindrop/{raindrop_id}"
        data = {"tags": tags}
        return self._make_request(endpoint, method="PUT", data=data)


class DeepSeekAPI:
    """DeepSeek AI API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
    
    def generate_tags(self, title: str, url: str, existing_tags: List[str] = None) -> List[str]:
        """Generate tags for a bookmark using DeepSeek"""
        
        existing_context = ""
        if existing_tags:
            existing_context = f"\nExisting tags: {', '.join(existing_tags)}"
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a bookmark tagging assistant. Generate 3-5 relevant, concise tags for bookmarks. Return ONLY the tags as a comma-separated list, nothing else. Tags should be lowercase, descriptive, and useful for categorization."
                },
                {
                    "role": "user",
                    "content": f"Generate tags for this bookmark:\nTitle: {title}\nURL: {url}{existing_context}"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        req = Request(
            self.base_url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        
        try:
            with urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                tags_text = data['choices'][0]['message']['content'].strip()
                # Remove any quotes or extra formatting
                tags_text = tags_text.replace('"', '').replace("'", "")
                tags = [t.strip().lower() for t in tags_text.split(',') if t.strip()]
                return tags
        except HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise Exception(f"DeepSeek API Error: {e.code} - {error_body}")


def build_collection_tree(root_collections: List[Dict], child_collections: List[Dict]) -> Dict:
    """Build a tree structure of collections with their children"""
    
    # Create a map of parent_id -> list of children
    children_map = {}
    for child in child_collections:
        parent = child.get("parent")
        if parent:
            # Handle both dict and direct ID formats
            if isinstance(parent, dict):
                parent_id = parent.get("$id")
            else:
                parent_id = parent
            
            if parent_id:
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(child)
    
    def add_children(collection: Dict) -> Dict:
        """Recursively add children to a collection"""
        coll_id = collection["_id"]
        collection["children"] = children_map.get(coll_id, [])
        for child in collection["children"]:
            add_children(child)
        return collection
    
    # Add children to root collections
    tree = [add_children(coll) for coll in root_collections]
    return tree


def print_collection_tree(collections: List[Dict], indent: int = 0):
    """Print collection tree structure"""
    prefix = "  " * indent
    for coll in collections:
        count = coll.get("count", 0)
        title = coll.get("title", "Untitled")
        coll_id = coll["_id"]
        print(f"{prefix}📁 {title} (ID: {coll_id}, {count} items)")
        
        if "children" in coll and coll["children"]:
            print_collection_tree(coll["children"], indent + 1)


def collect_all_collection_ids(collections: List[Dict]) -> List[int]:
    """Recursively collect all collection IDs from tree"""
    ids = []
    for coll in collections:
        ids.append(coll["_id"])
        if "children" in coll and coll["children"]:
            ids.extend(collect_all_collection_ids(coll["children"]))
    return ids


def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def process_raindrops(
    raindrop_api: RaindropAPI,
    deepseek_api: DeepSeekAPI,
    collection_ids: List[int],
    dry_run: bool = False,
    process_nested: bool = False
):
    """Process all raindrops in given collections"""
    
    total_raindrops = 0
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    start_time = time.time()
    
    for coll_idx, coll_id in enumerate(collection_ids, 1):
        # Calculate ETA
        if coll_idx > 1:
            elapsed = time.time() - start_time
            avg_time_per_collection = elapsed / (coll_idx - 1)
            remaining_collections = len(collection_ids) - coll_idx + 1
            eta_seconds = avg_time_per_collection * remaining_collections
            eta_str = format_time(eta_seconds)
            progress_pct = ((coll_idx - 1) / len(collection_ids)) * 100
            
            print(f"\n{'=' * 60}")
            print(f"Collection {coll_idx}/{len(collection_ids)} (ID: {coll_id}) | Progress: {progress_pct:.1f}% | ETA: {eta_str}")
            print('=' * 60)
        else:
            print(f"\n{'=' * 60}")
            print(f"Processing Collection {coll_idx}/{len(collection_ids)} - ID: {coll_id}")
            print('=' * 60)
        
        page = 0
        while True:
            # Get raindrops for this collection
            try:
                result = raindrop_api.get_raindrops(coll_id, page=page, perpage=50, nested=process_nested)
            except Exception as e:
                print(f"⚠️  Error fetching raindrops from collection {coll_id}: {e}")
                break
            
            items = result.get("items", [])
            if not items:
                if page == 0:
                    print(f"    📭 Empty collection")
                break
            
            for raindrop in items:
                total_raindrops += 1
                raindrop_id = raindrop["_id"]
                title = raindrop.get("title", "Untitled")
                link = raindrop.get("link", "")
                existing_tags = raindrop.get("tags", [])
                
                # Calculate per-raindrop ETA (after processing at least 10 items)
                if total_raindrops > 10:
                    elapsed = time.time() - start_time
                    avg_time_per_raindrop = elapsed / total_raindrops
                    
                    # Estimate remaining: items needing tags in current collection + all items in remaining collections
                    # This is a rough estimate - we don't know how many will be skipped
                    remaining_in_page = len(items) - items.index(raindrop)
                    remaining_collections = len(collection_ids) - coll_idx
                    estimated_remaining = remaining_in_page + (remaining_collections * 25)  # assume avg 25 items per collection
                    
                    eta_seconds = avg_time_per_raindrop * estimated_remaining
                    eta_str = format_time(eta_seconds)
                    
                    print(f"\n[{total_raindrops}] {title[:60]}... | ETA: {eta_str}")
                else:
                    print(f"\n[{total_raindrops}] {title[:70]}...")
                
                print(f"    URL: {link[:80]}...")
                
                # Skip if already has 3+ tags
                if len(existing_tags) >= 3:
                    skipped_count += 1
                    print(f"    ⊘ Skipped (already has {len(existing_tags)} tags: {', '.join(existing_tags)})")
                    continue
                
                # Generate tags
                try:
                    if dry_run:
                        # Mock tags for dry run
                        new_tags = ["tag1", "tag2", "tag3"]
                        print(f"    ✓ Tags (simulated): {', '.join(new_tags)}")
                    else:
                        new_tags = deepseek_api.generate_tags(title, link, existing_tags)
                        print(f"    🤖 Generated tags: {', '.join(new_tags)}")
                        
                        # Merge with existing tags and remove duplicates
                        merged_tags = list(set(existing_tags + new_tags))
                        
                        # Update raindrop
                        raindrop_api.update_raindrop(raindrop_id, merged_tags)
                        print(f"    ✓ Updated with tags: {', '.join(merged_tags)}")
                        
                        # Small delay between DeepSeek API calls
                        time.sleep(0.3)
                    
                    success_count += 1
                        
                except Exception as e:
                    error_count += 1
                    print(f"    ✗ Error: {str(e)}")
            
            # Move to next page
            page += 1
            
            # Check if there are more pages
            if page >= 100:  # Safety limit
                print(f"⚠️  Reached page limit for collection {coll_id}")
                break
        
        # Small delay between collections
        if coll_idx < len(collection_ids):
            time.sleep(0.5)
    
    return total_raindrops, success_count, error_count, skipped_count


def main():
    # Parse arguments
    dry_run = False
    collection_id = None
    process_nested = False
    args = sys.argv[1:]
    
    if '--dry-run' in args:
        dry_run = True
        args.remove('--dry-run')
    
    if '--nested' in args:
        process_nested = True
        args.remove('--nested')
    
    if '--collection-id' in args:
        idx = args.index('--collection-id')
        if idx + 1 < len(args):
            try:
                collection_id = int(args[idx + 1])
                args.pop(idx)  # Remove --collection-id
                args.pop(idx)  # Remove the ID value
            except ValueError:
                print("❌ Error: Invalid collection ID")
                sys.exit(1)
        else:
            print("❌ Error: --collection-id requires an ID value")
            sys.exit(1)
    
    if len(args) != 2:
        print("Usage: python raindrop_tagger.py <raindrop_token> <deepseek_api_key> [--dry-run] [--collection-id ID] [--nested]")
        print("\nExamples:")
        print("  python raindrop_tagger.py rdp_xxxxx sk-xxxxx --dry-run")
        print("  python raindrop_tagger.py rdp_xxxxx sk-xxxxx")
        print("  python raindrop_tagger.py rdp_xxxxx sk-xxxxx --collection-id 12345")
        print("  python raindrop_tagger.py rdp_xxxxx sk-xxxxx --collection-id 12345 --nested")
        print("\nOptions:")
        print("  --dry-run          Preview and simulate tagging without using API quota")
        print("  --collection-id    Process only a specific collection")
        print("  --nested           Include nested collections when using --collection-id")
        print("\nTo get your Raindrop token:")
        print("  1. Go to https://app.raindrop.io/settings/integrations")
        print("  2. Create new app or use existing")
        print("  3. Copy the 'Test token'")
        print("\nNote: The script respects Raindrop's 120 requests/minute rate limit.")
        print("      Processing 4000 bookmarks will take approximately 30-45 minutes.")
        sys.exit(1)
    
    raindrop_token = args[0]
    deepseek_key = args[1]
    
    mode_label = "🔍 DRY RUN MODE - No changes will be made" if dry_run else "🔴 LIVE MODE - Changes will be saved"
    print("🏷️  Raindrop.io AI Tagger (Direct API)")
    print("=" * 60)
    print(f"{mode_label}")
    print("=" * 60)
    
    # Initialize APIs
    raindrop_api = RaindropAPI(raindrop_token)
    deepseek_api = DeepSeekAPI(deepseek_key)
    
    # Fetch collections
    print("\n📚 Fetching collections...")
    try:
        root_collections = raindrop_api.get_root_collections()
        child_collections = raindrop_api.get_child_collections()
        
        print(f"✓ Found {len(root_collections)} root collections")
        print(f"✓ Found {len(child_collections)} nested collections")
        
        # Build collection tree
        collection_tree = build_collection_tree(root_collections, child_collections)
        
        # Print structure
        print("\n📂 Collection Structure:")
        print("-" * 60)
        print_collection_tree(collection_tree)
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Error fetching collections: {e}")
        sys.exit(1)
    
    # Determine which collections to process
    if collection_id:
        nested_msg = " (including nested)" if process_nested else ""
        print(f"\n🎯 Processing specific collection ID: {collection_id}{nested_msg}")
        collection_ids = [collection_id]
    else:
        print(f"\n🎯 Processing all collections")
        collection_ids = collect_all_collection_ids(collection_tree)
    
    print(f"📊 Will process {len(collection_ids)} collection(s)")
    
    # Calculate rough time estimate
    estimated_minutes = len(collection_ids) * 0.5  # rough estimate
    print(f"⏱️  Estimated time: ~{estimated_minutes:.0f} minutes (accounting for rate limits)")
    
    if not dry_run:
        confirm = input("\n⚠️  This will modify your Raindrop.io bookmarks. Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cancelled.")
            sys.exit(0)
    
    # Process raindrops
    action_word = "Simulating" if dry_run else "Processing"
    print(f"\n🤖 {action_word} raindrops with AI...")
    print(f"💡 Rate limiting is enabled - the script will pace itself automatically")
    
    start_time = time.time()
    
    total, success, errors, skipped = process_raindrops(
        raindrop_api,
        deepseek_api,
        collection_ids,
        dry_run,
        process_nested
    )
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"  Total raindrops processed: {total}")
    print(f"  Skipped (already tagged): {skipped}")
    print(f"  Successfully tagged: {success}")
    print(f"  Errors: {errors}")
    print(f"  Time elapsed: {elapsed_time/60:.1f} minutes")
    
    if dry_run:
        print(f"\n✅ Dry run complete! No API quota was used.")
        print("   To tag for real, remove the --dry-run flag.")
    else:
        print(f"\n✅ Done! Your bookmarks have been tagged in Raindrop.io")
        print("   Check your collections at https://app.raindrop.io")


if __name__ == "__main__":
    main()
