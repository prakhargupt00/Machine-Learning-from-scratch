#initialisation 
#Declaring the termination conditions for the K-means Alogrithm
# Tolerance is the minimum change in clusteroid vector required to continue with the next iteration.If change is less than
#tolerance we stop the algorithm.
tolerance = 0.0001
#The maximum number of iterations allowed.
max_iterations = 500
#dividing the dataset into upload and download data
upl_dataset = new_df[new_df['upload'] == 1]
dwn_dataset = new_df[new_df['download'] == 1]


import random 
# The function takes three inputs the state concerned , the type of data to cluster in that state i.e upload or download and the number of clusters.
# The function returns the centroids of each clusters along with all the points in that cluster. 

def K_means(state,upl_dwn,k):
    #picking attributes on which clustering is to be performed and the type of data(download/upload) 
    if upl_dwn == 'download':
        data = dwn_dataset[dwn_dataset[state] == 1][['Data_Speed.Kbps.','signal_strength','4G']]
    else:
        data = upl_dataset[upl_dataset[state] == 1][['Data_Speed.Kbps.','signal_strength','4G']]
    
     # randomly initializing the centroids
    centroids = []
    for i in range(k):
        index = int(random.uniform(0,len(data)-1))
        centroids.append(data.iloc[index , :])
    
    classes = [] 
    for i in range(max_iterations):
        
        for i in range(k):
            classes.append([])
        
        for i in range(0,len(data)):
           # print(features)
            distances = []
            for centroid in centroids:
                # calculating the euclidean distance of the point from the centroid
                distances.append(np.linalg.norm(data.iloc[i,:]-centroid))
            # calculating the index of the centroid to which it is closest
            classification = distances.index(min(distances))
            #adding the point to the cluster corresponding to that centroid
            classes[classification].append(data.iloc[i,:])

        previous = centroids
        #average the cluster datapoints to re-calculate the centroids
        for classification in range(0,len(classes)):
            centroids[classification] = np.average(classes[classification], axis = 0)
        
        #Assuming that the clustering is optimal
        isOptimal = True
        
        # Checking the tolerance condition i.e if the change between new clusteriod vector and the previous clusteroid vector is above tolerance
        # we see this as unoptimal solution(isOptimal=False) and continue to next iteration.
        
        for centroid in range(0,len(centroids)):
            original_centroid = previous[centroid]
            curr = centroids[centroid]
            if np.sum((curr - original_centroid)/original_centroid * 100.0) > tolerance:
                isOptimal = False
                
        #break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
        if isOptimal:
            break
    return classes,centroids    