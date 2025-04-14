import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("LV7 Grupiranje podataka primjenom algoritma K srednjih vrijednosti-20250412\imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255
# print(img.astype(np.float64)) 

# transformiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape # (601, 1195, 3)

img_array = np.reshape(img, (w*h, d)) # 718.195 redaka i 3 stupca
# reshape - gives a new shape to an array without changing its data
# print(img_array) svaki piksel je jedan redak

# rezultatna slika
img_array_aprox = img_array.copy()
# print(img_array_aprox)


#############################################################################
# ZAD 1

unique_colors = np.unique(img_array_aprox, axis=0)
print("Koliko je razlicitih boja prisutno u ovoj slici? ", len(unique_colors))
# da traži jedinstvene redove u 2D polju sto zapravo znaci jedinstvene PIKSELE

#############################################################################
# ZAD 2 - originalna slika

km = KMeans(n_clusters=5, init="k-means++", n_init=5, random_state=0)
km.fit(img_array)
# treniramo model, biramo pocetne centre, 
# iterativno premjesta centre da minimizira razliku, zavrsi kad su centri stabilni

labels = km.predict(img_array)
# svaki redak dobije oznaku kao kojoj grupi pripada
# print(labels) njih je hrpa

# labels polje brojeva koji predstavljaju grupu da svaki piksel
# [4, 4, 4, 4, 4, 5, 5, 6, 2, 1, ...]

centers = km.cluster_centers_ # 5 centara koji imaju RGB vrijednosti
# print(centers)

#############################################################################
# ZAD 3

img_array_aprox = centers[labels] # postavljamo boju centra 

# rekonstrukcija slike
img_aprox = np.reshape(img_array_aprox, (w, h, d)) # stvori sliku

plt.figure()
plt.imshow(img_aprox)
plt.show()

#############################################################################
# ZAD 4

# for K in [2, 4, 8, 16, 32]:
#     kmeans = KMeans(n_clusters=K, random_state=0)
#     kmeans.fit(img_array)
#     labels = kmeans.predict(img_array)
#     centers = kmeans.cluster_centers_
#     img_array_aprox = centers[labels]
#     img_aprox = np.reshape(img_array_aprox, (w, h, d))

#     plt.figure()
#     plt.title(f"Aproksimirana slika za K={K}")
#     plt.imshow(img_aprox)
#     plt.tight_layout()
#     plt.show()

# gubi detalje jer su sve boje reducirane na samo nekoliko
# sličnost s originalnom slikom raste, ali i memorija i vrijeme izvođenja

#############################################################################
# ZAD 5
img = Image.imread("LV7 Grupiranje podataka primjenom algoritma K srednjih vrijednosti-20250412\imgs\\test_3.jpg")
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()
img = img.astype(np.float64) / 255

w, h, d = img.shape
img_array = np.reshape(img, (w*h ,d))
img_array_aprox = img_array.copy()

for K in [2, 4, 8, 16, 32]:
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(img_array)
    labels = kmeans.predict(img_array)
    centers = kmeans.cluster_centers_
    img_array_aprox = centers[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))

    plt.figure()
    plt.title(f"Aproksimirana slika za K={K}")
    plt.imshow(img_aprox)
    plt.tight_layout()
    plt.show()

#############################################################################
# ZAD 6

# inertias = []
# K_values = list(range(1, 10))

# for K in K_values:
#     kmeans = KMeans(n_clusters=K, random_state=42)
#     kmeans.fit(img_array)
#     inertias.append(kmeans.inertia_)

# plt.figure()
# plt.plot(K_values, inertias, marker='o')
# plt.title("Elbow metoda: Inercija vs Broj klastera")
# plt.xlabel("Broj klastera (K)")
# plt.ylabel("Inercija")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


#############################################################################
# ZAD 7

# K = 5  # možeš birati bilo koji broj
# kmeans = KMeans(n_clusters=K, random_state=0).fit(img_array)
# labels = kmeans.labels_

# # Vizualizacija maski po klasterima
# for i in range(K):
#     # binarna maska za klaster i
#     mask = (labels == i).astype(np.uint8)
#     mask_image = np.reshape(mask, (w, h))  # vrati u oblik slike

#     plt.figure()
#     plt.title(f"Binarna slika za klaster {i}")
#     plt.imshow(mask_image, cmap='gray')  # crno-bijela
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()







