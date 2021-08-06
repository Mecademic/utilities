import numpy as np
from numpy import linalg as LA

class Mat():
    '''
    A Matrix class to handle the pose transformation and rotation matrices
    used for calculating WRF and TCP
    '''
    
    def __init__(self,pose):

        self.x = pose[0]
        self.y = pose[1]
        self.z = pose[2]
        
        alpha = np.deg2rad(pose[3])
        beta = np.deg2rad(pose[4])
        gamma = np.deg2rad(pose[5]) 
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)

        sin_beta = np.sin(beta)
        cos_beta = np.cos(beta)

        sin_gamma = np.sin(gamma)
        cos_gamma = np.cos(gamma)

        self.r11 = cos_beta*cos_gamma
        self.r12 = -(cos_beta)*(sin_gamma)
        self.r13 = sin_beta

        self.r21 = cos_alpha*sin_gamma + sin_alpha*sin_beta*cos_gamma
        self.r22 = cos_alpha*cos_gamma - sin_alpha*sin_beta*sin_gamma
        self.r23 = -(sin_alpha*cos_beta)

        self.r31 = sin_alpha*sin_gamma - cos_alpha*sin_beta*cos_gamma
        self.r32 = sin_alpha*cos_gamma + cos_alpha*sin_beta*sin_gamma
        self.r33 = cos_alpha*cos_beta


    def H(self):
        return np.array([[self.r11, self.r12, self.r13, self.x],[self.r21, self.r22, self.r23, self.y],[self.r31, self.r32, self.r33, self.z],[0, 0, 0, 1]])
       
    def R33(self):
        return np.array([[self.r11, self.r12, self.r13],[self.r21, self.r22, self.r23],[self.r31, self.r32, self.r33]])


def xyzRxyz_to_H(pose):
    '''
    Function to calculate the Pose Transformation Matrix for a given (x,y,z,rx,ry,rz)
    ***Uses Euler Angle notation***
    '''
    return Mat(pose).H()

def H_to_xyzRxyz(Pose_Mat):
    '''
    Function to calculate the (x,y,z,rx,ry,rz) from the Pose Transormation Matrix.
    ***Uses Euler Angle notation***
    '''
    if abs(Pose_Mat[0,2]) == 1:
        ry = Pose_Mat[0,2]*90

        rz_rad = np.arctan2(Pose_Mat[1,0],Pose_Mat[1,1])
        rz = np.rad2deg(rz_rad)

        rx = 0
    else:
        ry_rad = np.arcsin(Pose_Mat[0,2])
        ry = np.rad2deg(ry_rad)

        rz_rad = np.arctan2(-Pose_Mat[0,1],Pose_Mat[0,0])
        rz = np.rad2deg(rz_rad)

        rx_rad = np.arctan2(-Pose_Mat[1,2],Pose_Mat[2,2])
        rx = np.rad2deg(rx_rad)

    return (Pose_Mat[0,3],Pose_Mat[1,3],Pose_Mat[2,3],rx,ry,rz)
        

def CalcWRFChild(WRF_Parent, WRF_Child):
    '''
    Function to calculate the WRF of a child reference frame with respect to BRF of Meca500
    Takes the parent WRF and child WRF values and returns the WRF of child with respect to BRF of Meca500
    Parent Frame: Reference frame defined with respect to BRF of Meca500
    Child Frame: Reference frame defined with respect to Parent Frame
    '''
    Pose_Parent = xyzRxyz_to_H(WRF_Parent)
    Pose_Child = xyzRxyz_to_H(WRF_Child)

    Pose_Child_Parent = Pose_Parent.dot(Pose_Child)

    return str(H_to_xyzRxyz(Pose_Child_Parent))


def CalcWRF(p1,p2,p3):
    '''
    Function to calculate the WRF with respect to BRF of Meca500 using 3 points
    P1 is at Origin of the reference frame
    P2 is a point on the +X axis of the reference frame
    P3 is a point on the +XY plane of the reference frame
    '''
    p1p2 = True
    p1p3 = True
    
    ux = [p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]] 
    norm_ux = LA.norm(ux)
    if norm_ux < 10:
        print(" P1 and P2 are too close to each other")
        p1p2 = False
        
    else:
        uvx = ux / norm_ux # Unit Vector along X

    if p1p2:
        v13 = [p3[0]-p1[0],p3[1]-p1[1],p3[2]-p1[2]] 
        norm_v13 = LA.norm(v13)
        if norm_v13 < 10:
            print(" P3 and P1 are too close to each other")
            p1p3 = False
            
        else:
            uv13 = v13 / norm_v13
    
    if p1p2 and p1p3:
        uz = np.cross(uvx,uv13)
        norm_uz = LA.norm(uz)
        uvz = uz / norm_uz # Unit Vector along Z

        uvy = np.cross(uvz,uvx) # Unit Vector along Y
        
        if abs(uvz[0]) == 1:
            beta_deg = uvz[0]*90

            gamma = np.arctan2(uvy[0],uvx[0])
            gamma_deg = np.rad2deg(gamma)

            alpha_deg = 0
    
        else:
            beta = np.arcsin(uvz[0])
            beta_deg = np.rad2deg(beta)

            gamma = np.arctan2(-uvy[0],uvx[0])
            gamma_deg = np.rad2deg(gamma)

            alpha = np.arctan2(-uvz[1],uvz[2])
            alpha_deg = np.rad2deg(alpha)

    
        result = [p1[0],p1[1],p1[2],alpha_deg,beta_deg,gamma_deg]
        #result_str = 'setWRF'+'(' + (','.join(format(vi, ".3f") for vi in result)+')')
        
    else:
        result= []
        #result_str = " Invalid Data"
    return result
        
def CalcTCP(nPoses):
    '''
    Function to calculate the TCP (X,Y,Z) from 4 or more poses
    Takes a list of Poses as input and returns the TCP 
    
        TCP(x,y,z) = [(R_Transpose.R)^-1].R_Transpose.p
    '''
    
    #nRows = 3*len(nPoses)
    #nCols = 3

    R = np.empty((0,3))
    p = np.empty((0,1))

    nPoseMat = []
    for pose in nPoses:
        nPoseMat.append(Mat(pose))

    for i in range(len(nPoses)):
        if i < len(nPoses)-1:
            R = np.append(R, np.array([[nPoseMat[i].r11 - nPoseMat[i+1].r11, nPoseMat[i].r12 - nPoseMat[i+1].r12, nPoseMat[i].r13 - nPoseMat[i+1].r13],[nPoseMat[i].r21 - nPoseMat[i+1].r21, nPoseMat[i].r22 - nPoseMat[i+1].r22, nPoseMat[i].r23 - nPoseMat[i+1].r23],[nPoseMat[i].r31 - nPoseMat[i+1].r31, nPoseMat[i].r32 - nPoseMat[i+1].r32, nPoseMat[i].r33 - nPoseMat[i+1].r33]]),axis = 0)
            p = np.append(p,np.array([[-nPoseMat[i].x + nPoseMat[i+1].x],[-nPoseMat[i].y + nPoseMat[i+1].y],[-nPoseMat[i].z + nPoseMat[i+1].z]]),axis = 0)
            
            
        elif i == len(nPoses)-1:
            R = np.append(R, np.array([[nPoseMat[i].r11 - nPoseMat[0].r11, nPoseMat[i].r12 - nPoseMat[0].r12, nPoseMat[i].r13 - nPoseMat[0].r13],[nPoseMat[i].r21 - nPoseMat[0].r21, nPoseMat[i].r22 - nPoseMat[0].r22, nPoseMat[i].r23 - nPoseMat[0].r23],[nPoseMat[i].r31 - nPoseMat[0].r31, nPoseMat[i].r32 - nPoseMat[0].r32, nPoseMat[i].r33 - nPoseMat[0].r33]]),axis = 0)
            p = np.append(p,np.array([[-nPoseMat[i].x + nPoseMat[0].x],[-nPoseMat[i].y + nPoseMat[0].y],[-nPoseMat[i].z + nPoseMat[0].z]]),axis = 0)
            
    
    TCP = (LA.inv(R.transpose().dot(R))).dot(R.transpose()).dot(p)
    return TCP   

