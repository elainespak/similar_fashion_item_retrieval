# Most similar product images retrieval
### Process
1. Remove background (**check out [Cloudinary API](https://cloudinary.com/pricing)**)
2. Crop product image (maybe unnecessary)
3. Extract feature vector (feature_vector.py)
4. Calculate cosine similarity (similar_items.py; **look into other similarity metrics**)
5. Grab and display similar products (similar_items.py)
