"""
Mock data generation for testing and demonstration
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class MockDataGenerator:
    """Generate realistic mock review data"""
    
    def __init__(self):
        self.positive_templates = [
            "Amazing {product}! Highly recommend it. Great quality and fast delivery.",
            "Love this {product}! Exceeded my expectations. Will buy again.",
            "Excellent {product}. Perfect for my needs. 5 stars!",
            "Outstanding quality {product}. Worth every penny. Fast shipping too.",
            "Best {product} I've ever bought. Incredible value for money.",
            "Fantastic {product}! Works perfectly. Great customer service.",
            "Superb {product}. Exactly as described. Very satisfied.",
            "Top-notch {product}. Highly durable and well-made.",
            "Incredible {product}! Solved all my problems. Recommended!",
            "Perfect {product}. Great design and functionality.",
            "Wonderful {product}. Easy to use and very effective.",
            "Brilliant {product}! Couldn't be happier with this purchase.",
            "Exceptional {product}. High quality materials and construction.",
            "Awesome {product}! Works better than expected.",
            "Great {product}. Fast delivery and excellent packaging."
        ]
        
        self.negative_templates = [
            "Terrible {product}. Broke after one week. Waste of money.",
            "Poor quality {product}. Not as advertised. Very disappointed.",
            "Awful {product}. Doesn't work properly. Returning it.",
            "Bad {product}. Cheap materials and poor construction.",
            "Horrible {product}. Completely useless. Don't buy this.",
            "Disappointing {product}. Not worth the price. Poor quality.",
            "Defective {product}. Stopped working immediately. Terrible.",
            "Worst {product} ever. Complete waste of time and money.",
            "Faulty {product}. Multiple issues from day one.",
            "Useless {product}. Nothing like the description. Avoid!",
            "Broken {product}. Arrived damaged and doesn't work.",
            "Overpriced {product}. Poor quality for the money.",
            "Unreliable {product}. Keeps malfunctioning.",
            "Cheap {product}. Falls apart easily. Very poor build quality.",
            "Defective {product}. Had to return it immediately."
        ]
        
        self.neutral_templates = [
            "Okay {product}. Does the job but nothing special.",
            "Average {product}. Some good points, some bad.",
            "Decent {product}. Works as expected. Nothing extraordinary.",
            "Fair {product}. Could be better but acceptable.",
            "Standard {product}. Gets the job done adequately.",
            "Mediocre {product}. Has pros and cons.",
            "Reasonable {product}. Not great, not terrible.",
            "Acceptable {product}. Meets basic requirements.",
            "So-so {product}. Mixed feelings about this purchase.",
            "Regular {product}. Nothing to complain about, nothing to praise.",
            "Basic {product}. Does what it's supposed to do.",
            "Ordinary {product}. Neither impressed nor disappointed.",
            "Standard {product}. Average quality for the price.",
            "Typical {product}. What you'd expect for this price range.",
            "Moderate {product}. Some features work well, others don't."
        ]
        
        self.reviewer_names = [
            "John D.", "Sarah M.", "Mike R.", "Lisa K.", "David W.",
            "Emma S.", "Chris P.", "Anna L.", "Tom B.", "Maria G.",
            "Alex J.", "Sophie T.", "Ryan H.", "Kate F.", "Mark C.",
            "Jessica R.", "Daniel M.", "Amy N.", "Steve L.", "Rachel W.",
            "Kevin S.", "Laura B.", "James T.", "Nicole P.", "Brian K.",
            "Michelle D.", "Andrew F.", "Stephanie H.", "Robert G.", "Jennifer M."
        ]
    
    def generate_sample_reviews(self, product_name, num_reviews=50):
        """Generate sample reviews for a product"""
        reviews = []
        
        # Realistic sentiment distribution
        # 45% positive, 25% negative, 30% neutral
        sentiment_distribution = ['positive'] * 45 + ['negative'] * 25 + ['neutral'] * 30
        random.shuffle(sentiment_distribution)
        
        # Ensure we have enough sentiments
        while len(sentiment_distribution) < num_reviews:
            sentiment_distribution.extend(['positive'] * 45 + ['negative'] * 25 + ['neutral'] * 30)
        
        sentiment_distribution = sentiment_distribution[:num_reviews]
        
        for i in range(num_reviews):
            sentiment = sentiment_distribution[i]
            
            # Select template based on sentiment
            if sentiment == 'positive':
                template = random.choice(self.positive_templates)
                rating = random.choices([4, 5], weights=[30, 70])[0]
            elif sentiment == 'negative':
                template = random.choice(self.negative_templates)
                rating = random.choices([1, 2], weights=[60, 40])[0]
            else:  # neutral
                template = random.choice(self.neutral_templates)
                rating = 3
            
            review_text = template.format(product=product_name)
            
            # Add some variation
            variations = [
                f" The delivery was {'fast' if rating >= 4 else 'slow'}.",
                f" Customer service was {'helpful' if rating >= 4 else 'poor'}.",
                f" Packaging was {'excellent' if rating >= 4 else 'damaged'}.",
                f" Would {'definitely' if rating >= 4 else 'not'} recommend.",
                f" Price is {'reasonable' if rating >= 3 else 'too high'}."
            ]
            
            if random.random() < 0.4:  # 40% chance to add variation
                review_text += random.choice(variations)
            
            # Generate random date within last 6 months
            days_ago = random.randint(1, 180)
            review_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            reviews.append({
                'review_id': f"REV_{i+1:03d}",
                'reviewer_name': random.choice(self.reviewer_names),
                'rating': rating,
                'review_text': review_text,
                'date': review_date,
                'verified_purchase': random.choices([True, False], weights=[80, 20])[0],
                'helpful_votes': random.randint(0, 50) if random.random() < 0.3 else 0
            })
        
        return pd.DataFrame(reviews)
    
    def generate_product_info(self, product_name):
        """Generate mock product information"""
        categories = [
            "Electronics", "Home & Garden", "Sports & Outdoors",
            "Books", "Clothing", "Health & Beauty", "Toys & Games",
            "Automotive", "Tools & Hardware", "Kitchen & Dining"
        ]
        
        return {
            'product_name': product_name,
            'category': random.choice(categories),
            'price': round(random.uniform(10, 500), 2),
            'average_rating': round(random.uniform(2.5, 4.8), 1),
            'total_reviews': random.randint(50, 1000),
            'availability': random.choice(['In Stock', 'Limited Stock', 'Out of Stock'])
        }