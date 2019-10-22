from matplotlib import pyplot as plt

'''
connected component labeling
developed by Cong Zou, 10/2/2019
'''

class CCLfunc(object):
    def __init__(self, image):
        #super(CCLfunc, self).__init__()
        self.img = image
        self.height = len(self.img)
        self.width = len(self.img[0])
        self.E_table = []
        

        self.C_table = {}
        self.colorcnt = 0
        
    #sequential manipulation using label to label pixels and build E_table first time
    def Sequential(self, img):
        for i in range(self.height):
            for j in range(self.width):
                if img[i][j][0] > 0:
                    if i == 0 and j == 0 :
                        self.Label(0, 0, [i, j])
                    elif i == 0 and j != 0:
                        self.Label(img[i][j-1][0], 0, [i, j])
                    elif i != 0 and j == 0:
                        self.Label(0, img[i-1][j][0], [i, j])
                    else:
                        self.Label(img[i][j-1][0], img[i-1][j][0], [i, j])
        return img
        
    #used in sequential function to complete first scan and label of image
    def Label(self, left, up, position):
        if left == 0 and up == 0:
            self.colorcnt += 1
            self.E_table.append(len(self.E_table) + 1)
            self.img[position[0]][position[1]][0] = self.E_table[-1]
            self.img[position[0]][position[1]][1:3] = 0
        elif left != 0 and up != 0 and left != up:
            self.img[position[0]][position[1]][0] = min(left, up)
            self.img[position[0]][position[1]][1:3] = 0
            self.E_table = self.Buildetable(left, up, self.E_table)
        else:
            self.img[position[0]][position[1]][0] = max(left, up)
            self.img[position[0]][position[1]][1:3] = 0
            
    #During first scan of image, Build the E_table
    def Buildetable(self, left, up, E_table):
        while(left != E_table[left - 1]):
            left = E_table[left - 1]
        while(up != E_table[up - 1]):
            up = E_table[up - 1]
        if up != left:
            E_table[max(left, up) - 1] = min(left, up)
            self.colorcnt -= 1
        return E_table
    
    #rearrange and complete E_table here
    def Rearrangeetable(self, E_table):
        for i in range (len(E_table)):
            if(i != E_table[i] - 1):
                temp = i
                while(temp != E_table[temp] - 1):
                    temp = E_table[temp] - 1
                E_table[i] = E_table[temp]
        return E_table
                
    #Build a dictionary of color table, [:3] is the pixel color corresponding to key value
    #C_table.values[3] records the pixels' number which can be applied to filter
    def Buildctable(self, E_table, C_table):
        param = ((192*3)/(self.colorcnt+1))
        count = 1
        
        
        for num in E_table:
            if C_table.get(num) == None:
                color = [1,0,0,0]
                count += 1
                temp = param*count
                
                if temp <= 191:
                    color[0] = temp + 31
                elif temp > 191 and temp <= 381:
                    color[1] = temp - 161
                elif temp > 381:
                    color[2] = temp - 350
                else:
                    raise Exception("The value of E_table must be less than colorcnt")
                C_table[num] = color
        return C_table

    #size filter function, threshold can be defined
    def Size_filter(self, threshold, E_table, C_table):
        for i in range(self.height):
            for j in range(self.width):
                if self.img[i][j][0] != 0:
                    C_table[E_table[self.img[i][j][0] - 1]][-1] += 1
                    
        
        for color in C_table.values():
            if(color[-1] < threshold):
                for i in range(3):
                    color[i] = 0
        return C_table
                    
    #second scan of image, fill the pixel with color value assigned in C_table
    def Relabel(self, img, E_table, C_table):
        for i in range(self.height):
            for j in range(self.width):
                if img[i][j][0] != 0:
                    color = self.C_table[self.E_table[img[i][j][0] - 1]]
                    for k in range(3):
                        img[i][j][k] = color[k]
        return img
            
    #convenient use of CCL algo with filter
    def Filter_forward(self, threshold):
        img = self.Sequential(self.img)
        self.E_table = self.Rearrangeetable(self.E_table)
        C_t = self.Buildctable(self.E_table, self.C_table)
        C_t2 = self.Size_filter(threshold, self.E_table, self.C_table)
        img = self.Relabel(img, self.E_table, C_t2)
        return img
    
    #convenient use of CCL without filter
    def Forward(self):        
        img = self.Sequential(self.img)
        self.E_table = self.Rearrangeetable(self.E_table)
        C_t = self.Buildctable(self.E_table, self.C_table)
        img = self.Relabel(img, self.E_table, C_t)
        return img
'''
    def __call__(self):
        return forward()
'''




        










        





            
            
