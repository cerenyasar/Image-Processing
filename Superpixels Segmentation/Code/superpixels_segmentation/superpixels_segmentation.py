import sys
import cv2
from math import sqrt
import numpy as np
from operator import itemgetter

def initClusCenter(Clus,img,k):
    row=img.shape[0]
    col=img.shape[1]
    s=int(sqrt(row*col/k))
    print(s)
    c=[]
    t=int(s/2-1)
    c.append(t)
    c.append(t)
    c.append(img[t][t][0])
    c.append(img[t][t][1])
    c.append(img[t][t][2])
    Clus.append(c)
    k=0
    i=t
    j=i+s
    while i < row:
        while j < col:
            #print ("value: "+str(i)+" " +str(j))
            c=[]
            c.append(i)
            c.append(j)
            c.append(img[i][j][0])
            c.append(img[i][j][1])
            c.append(img[i][j][2])
            Clus.append(c)
            j+=s
        i+=s
        j=t
def bright_ave(New,Clus,img,l):
    row=img.shape[0]
    col=img.shape[1]
    for i in range(0,len(Clus)):
        sum_l=0
        sum_a=0
        sum_b=0
        sum=0
        ave_l=0
        ave_a=0
        ave_b=0
        j=0
        #print("len["+str(i)+"] :"+ str(len(l[i])))
        while j < len(l[i]):
            #print("j : "+str(j))
            x=l[i][j]
            y=l[i][j+1]
            sum_l+=img[x][y][0]
            sum_a+=img[x][y][1]
            sum_b+=img[x][y][2]
            sum+=1
            j+=2
        ave_l=sum_l // sum
        ave_a=sum_a //sum
        ave_b=sum_b //sum
        j=0
        while j < len(l[i]):
            
            x=l[i][j]
            y=l[i][j+1]
            #print("ilk L a b : " +str(img[x][y][0])+" " +str(img[x][y][1])+" "+str(img[x][y][2]))
            img[x][y][0]=ave_l
            img[x][y][1]=ave_a
            img[x][y][2]=ave_b
            #print("ortalama L a b : " +str(img[x][y][0])+" " +str(img[x][y][1])+" "+str(img[x][y][2]))
            j+=2
    

def newClusCen(New,Clus,img,l):
    row=img.shape[0]
    col=img.shape[1]
    for i in range(0,len(Clus)):
        sum_row=0
        sum_col=0
        sum_r=0
        sum_g=0
        sum_b=0
        sum=0
        newCenter=[]
        j=0
        #print("len["+str(i)+"] :"+ str(len(l[i])))
        while j < len(l[i]):
            #print("j : "+str(j))
            x=l[i][j]
            y=l[i][j+1]
            sum_row+=x
            sum_col+=y
            sum_r+=img[x][y][0]
            sum_g+=img[x][y][1]
            sum_b+=img[x][y][2]
            sum+=1
            j+=2


        """for x in range(0,row):
            for y in range(0,col):
                if(l[x][y]==i):
                    sum_row+=x
                    sum_col+=y
                    sum_r+=img[x][y][0]
                    sum_g+=img[x][y][1]
                    sum_b+=img[x][y][2]
                    sum+=1
        """
        newCenter.append(sum_row//sum)
        newCenter.append(sum_col//sum)
        newCenter.append(sum_r//sum)
        newCenter.append(sum_g//sum)
        newCenter.append(sum_b//sum)
        New.append(newCenter)






set=["10100_11400","14000_21400","20500_10700","29200_21400","31300_16200"]

def superPix(string,k_pix):
    for element in set:
        fileName= element + string
        try:
            fout=open(fileName,"rb")
        except:
            print ("Cannot open file ", filename, "Exiting … \n")
            sys.exit()
        img = cv2.imread(fileName)
        Clus=[]
        initClusCenter(Clus,img,k_pix)
        
        row=img.shape[0]
        col=img.shape[1]
        s=int(sqrt(row*col/k_pix))
        #initialize distance and label for every pixel
        l=[]
        d=[]
        
        for i in range(0,row):
            
            dtuple=[]
            for j in range(0,col):
                #ltuple.append(-1)
                dtuple.append(sys.maxsize)
            #l.append(ltuple)
            d.append(dtuple)
        m=40
        control=True
        while control==True:

            print(len(Clus))

            for k in range(0,len(Clus)):
                #print("Clus["+ str(k)+"][0] " + str(Clus[k][0]))
                #print("Clus["+ str(k)+"][1] " + str(Clus[k][1]))
                if Clus[k][0]-s < 0 :
                    stRow=0
                else:
                    stRow=Clus[k][0]-s
                if Clus[k][0]+s > row:
                    endRow=row
                else:
                    endRow=Clus[k][0]+s
                if Clus[k][1]-s < 0 :
                    stCol=0
                else:
                    stCol=Clus[k][1]-s
                if Clus[k][1]+s > col:
                    endCol=col
                else:
                    endCol=Clus[k][1]+s
                ls=[]
                for i in range(stRow,endRow):#satırlar
                    for j in range(stCol,endCol):#sütunlar
                        #print("k "+ str(k) + " i "+ str(i)+ " j " +str(j))
                        dc=sqrt((Clus[k][2]-img[i][j][0])**2 + (Clus[k][3]-img[i][j][1])**2 + (Clus[k][4]-img[i][j][2])**2)
                        dxy=((Clus[k][0]-i)**2 + (Clus[k][1]-j)**2)**0.5
                        ds=((dc)**2 + (((dxy/s)**2)*(m)**2))**0.5
                        if( ds < d[i][j]):
                            d[i][j]=ds
                            ls.append(i)
                            ls.append(j)
                l.append(ls)
            New=[]
            newClusCen(New,Clus,img,l)
            result=0
            for i in range(0,len(Clus)):
                result+= abs((Clus[i][0]-New[i][0])+(Clus[i][1]-New[i][1])) 
                #print("result "+ str(result)) 
            print("result "+ str(result))  
            if result < 10:
                control= False
            else:
                for i in range(0,len(Clus)):
                    Clus[i]=New[i]
            
        Lab_img= cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        Lab_l=[]
        for i in range(0,row):
            for j in range(0,col):
                d[i][j]=sys.maxsize
            
            
        control=True
        while control==True:
            print(len(Clus))

            for k in range(0,len(Clus)):
                #print("Clus["+ str(k)+"][0] " + str(Clus[k][0]))
                #print("Clus["+ str(k)+"][1] " + str(Clus[k][1]))
                if Clus[k][0]-s < 0 :
                    stRow=0
                else:
                    stRow=Clus[k][0]-s
                if Clus[k][0]+s > row:
                    endRow=row
                else:
                    endRow=Clus[k][0]+s
                if Clus[k][1]-s < 0 :
                    stCol=0
                else:
                    stCol=Clus[k][1]-s
                if Clus[k][1]+s > col:
                    endCol=col
                else:
                    endCol=Clus[k][1]+s
                ls=[]
                for i in range(stRow,endRow):#satırlar
                    for j in range(stCol,endCol):#sütunlar
                        #print("k "+ str(k) + " i "+ str(i)+ " j " +str(j))
                        dc=sqrt((Clus[k][2]-Lab_img[i][j][0])**2 + (Clus[k][3]-Lab_img[i][j][1])**2 + (Clus[k][4]-Lab_img[i][j][2])**2)
                        dxy=((Clus[k][0]-i)**2 + (Clus[k][1]-j)**2)**0.5
                        ds=((dc)**2 + (((dxy/s)**2)*(m)**2))**0.5
                        if( ds < d[i][j]):
                            d[i][j]=ds
                            ls.append(i)
                            ls.append(j)
                Lab_l.append(ls)
            New=[]
            newClusCen(New,Clus,Lab_img,Lab_l)
            result=0
            for i in range(0,len(Clus)):
                result+= abs((Clus[i][0]-New[i][0])+(Clus[i][1]-New[i][1])) 
                #print("result "+ str(result)) 
            print("result "+ str(result))  
            if result < 5:
                control= False
            else:
                for i in range(0,len(Clus)):
                    Clus[i]=New[i]
        bright_ave(New,Clus,Lab_img,Lab_l)
        RGB_img= cv2.cvtColor(Lab_img, cv2.COLOR_LAB2RGB)
        FileNameNew= element +"_" +str(m)+ string 
        cv2.imwrite(FileNameNew,RGB_img)
        #control new_centers==old_centers
        #np.array_equal(a,b) return false or true
        
print("Yapıyor")
superPix(".tiff",10000)