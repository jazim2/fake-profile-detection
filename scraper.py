import instaloader
import re
from urllib.parse import urlparse

def get_profile_data(username_or_url):
    """
    Fetches profile metadata using Instaloader.
    Returns a dictionary of features or None if failed.
    """
    L = instaloader.Instaloader()
    
    # Extract username if full URL is provided
    username = username_or_url
    if "instagram.com" in username_or_url:
        path = urlparse(username_or_url).path
        username = path.strip("/").split("/")[0]
    
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        
        # specific feature extraction matching our model's needs
        # Model expects: 
        # profile_pic (1/0), ratio_numlen_username, len_fullname, ratio_numlen_fullname,
        # sim_name_username (0-2), len_desc, extern_url (1/0), private (1/0), 
        # num_posts, num_followers, num_following
        
        data = {}
        
        # Basic Stats
        data['username'] = profile.username
        data['num_followers'] = profile.followers
        data['num_following'] = profile.followees
        data['num_posts'] = profile.mediacount
        data['private'] = 1 if profile.is_private else 0
        data['profile_pic'] = 1 if not profile.is_private else 0 # Assumption: public profiles usually have pics visible, privacy might hide it in some contexts but usually it's there. 
        # Better check: profile.profile_pic_url is almost always present even for default. 
        # Let's assume 1 unless it looks like a default placeholder (hard to check without downloading).
        # For the model 'profile_pic' feature (Yes/No), we'll set 1.
        
        data['len_desc'] = len(profile.biography) if profile.biography else 0
        data['extern_url'] = 1 if profile.external_url else 0
        
        # Advanced Text Features
        data['len_fullname'] = len(profile.full_name) if profile.full_name else 0
        
        num_in_username = len(re.findall(r'\d', profile.username))
        data['ratio_numlen_username'] = num_in_username / len(profile.username) if len(profile.username) > 0 else 0
        
        num_in_fullname = len(re.findall(r'\d', profile.full_name)) if profile.full_name else 0
        data['ratio_numlen_fullname'] = num_in_fullname / len(profile.full_name) if profile.full_name and len(profile.full_name) > 0 else 0
        
        # Similarity Name-Username
        # 0: No match, 1: Partial, 2: Full
        if profile.username.lower() == profile.full_name.lower().replace(" ", ""):
            data['sim_name_username'] = 2
        elif profile.username.lower() in profile.full_name.lower() or profile.full_name.lower() in profile.username.lower():
            data['sim_name_username'] = 1
        else:
            data['sim_name_username'] = 0
            
        return data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
