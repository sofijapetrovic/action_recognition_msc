from enum import Enum
import cv2


class JointType(Enum):
    LEFT_HAND = 0
    LEFT_ELBOW = 1
    LEFT_SHOULDER = 2
    HEAD = 3
    RIGHT_SHOULDER = 4
    RIGHT_ELBOW = 5
    RIGHT_HAND = 6

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class LimbType(Enum):
    LEFT_FOREARM = 0
    LEFT_ARM = 1
    LEFT_SHOULDER = 2
    RIGHT_SHOULDER = 3
    RIGHT_ARM = 4
    RIGHT_FOREARM = 5

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


LIMB_JOINT_DICT = {'left_forearm': (JointType.LEFT_HAND, JointType.LEFT_ELBOW),
              'left_arm': (JointType.LEFT_ELBOW, JointType.LEFT_SHOULDER),
              'left_shoulder': (JointType.LEFT_SHOULDER, JointType.HEAD),
              'right_shoulder': (JointType.HEAD, JointType.RIGHT_SHOULDER),
              'right_arm': (JointType.RIGHT_SHOULDER, JointType.RIGHT_ELBOW),
              'right_forearm': (JointType.RIGHT_ELBOW, JointType.RIGHT_HAND)}


def create_joints(pred_peaks, id_list=None):
    """
    Create joint objects from the estimator output
    Args:
        pred_peaks (list): list of lists

    Returns: list of joint objects

    """
    joints = []
    if id_list is not None:
        for j, joint_array in enumerate(pred_peaks):
            joints.append(Joint(x=joint_array[3], y=joint_array[2], type=JointType(joint_array[1]), id=id_list[j]))
    else:
        for joint_array in pred_peaks:
            joints.append(Joint(x=joint_array[3], y=joint_array[2], type=JointType(joint_array[1])))
    return joints


class Joint(object):
    def __init__(self, x, y, type, confidence=None, id=None):
        """
        Joint object constructor
        Args:
            x (pixels): x position
            y (pixels): y position
            type (JointType): joint type
        """
        self.x = x
        self.y = y
        self.type = type
        self.confidence = confidence
        self.person_id = id

    def __str__(self):
        return f'{self.type.name}{self.x, self.y}'

    def to_int(self):
        self.x = int(self.x)
        self.y = int(self.y)
    def draw(self, image):
        """
        Draws a joint as a circle on the image
        Args:
            image (np.array): Raw image
        """
        cv2.circle(image, (self.x, self.y), 20, (0,0,255), -1)


class Limb(object):
    def __init__(self, joint1, joint2, type):
        """
        Limb object constructor
        Args:
            joint1 (Joint): first joint of the limb
            joint2 (Joint): second joint of the limb
            type (LimbType): limb type
        """
        self.joint1 = joint1
        self.joint2 = joint2
        self.type = type

    def get_joints(self):
        """
        Get joints of the limb
        Returns: two joints

        """
        return self.joint1, self.joint2

    def contains_joint(self, joint):
        """
        Checks if one of the joints of the limb are the joint received as an argument to this method
        Args:
            joint (Joint): a joint which should be checked if it is a part of the limb

        Returns: Bool - true if joint belongs to the limb
        """
        return joint == self.joint1 or joint == self.joint2

    def draw(self, image):
        """
        Draws a limb as a line on the image and its containing joints as circles
        Args:
            image (np.array): raw image
        """
        self.joint1.to_int()
        self.joint2.to_int()
        cv2.line(image, (self.joint1.x, self.joint1.y), (self.joint2.x, self.joint2.y), (255, 0, 0), 10)
        self.joint1.draw(image)
        self.joint2.draw(image)




