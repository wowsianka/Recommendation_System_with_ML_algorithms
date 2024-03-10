# Recommendation_System_with_ML_algorithms
## Introduction:
Recommendation systems play a pivotal role in today's technology-dominated world, being applicable in areas such as e-commerce, social networks, search engines, and content promotion. Their primary aim is to personalize user experiences across various platforms by suggesting products, services, or information based on users' past interactions, preferences, and behavior. With the internet's growth and the digitalization of our daily lives, the vast amount of data generated presents new challenges and opportunities, including information overload. To address this, many online platforms utilize recommendation systems - intelligent algorithms that sift through vast data to suggest the most relevant items to users.


## The algorithms used in the research and development of the recommendation system included:

- Traditional Collaborative Filtering and Matrix Factorization: These are foundational techniques in recommendation systems. Collaborative filtering focuses on using the interactions and feedback of users to recommend items by finding similar users or items. Matrix factorization, on the other hand, decomposes the user-item interaction matrix into lower-dimensional matrices, revealing latent factors associated with users and items.

- Various Machine Learning Algorithms for Recommendation Engines: The text mentions starting with basic collaborative filtering and matrix factorization, and then moving to more advanced deep learning models. Specifically, algorithms such as Random System, BaselineOnly, and KNN With Means are mentioned as common in machine learning but showed weak results in this context, indicating a challenge in detecting patterns for effective movie recommendations.

- Deep Neural Networks (DNNs): Recognized for their capability to identify complex patterns and generalize data, neural networks were utilized to model complex user-item interactions.

- Non-negative Matrix Factorization (NMF): An advanced method that was evaluated and compared for its effectiveness in the recommendation system. NMF is used for identifying the latent relationships between users and items by decomposing the interaction matrix into two non-negative matrices, which can improve the quality of recommendations.

- A Novel Combination of Deep Neural Networks and Non-negative Matrix Factorization: The research culminated in a new recommendation system that successfully integrated deep neural networks with NMF to apply collaborative filtering based purely on user rating history, focusing on identifying and utilizing relationships between users to improve recommendation quality.

This multi-faceted approach, leveraging a combination of machine learning algorithms, aimed to tackle the complexities of recommendation systems and address the challenge of information overload by providing more accurate and personalized recommendations.

## Conclusion: 
This work focuses on exploring, implementing, and comparing various techniques used in recommendation systems, offering a comprehensive overview from basic concepts like collaborative filtering and content-based methods to advanced hybrid systems. An exploratory data analysis was conducted on a real movie dataset, leading to the development and comparison of several machine learning algorithms. Ultimately, a novel recommendation system combining deep neural networks with non-negative matrix factorization was presented, focusing on leveraging user rating histories to identify relationships. While the system achieved satisfactory results, there are opportunities for further optimization and development. Future research could include A/B testing in a real production environment to more accurately evaluate system functionality and user behavior. Additionally, enriching the system with more user features and advanced technologies could enhance recommendation effectiveness.
