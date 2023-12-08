import json
from typing import Dict, List, Set
import numpy as np
import os
from math import cos, sin
import yaml
from yaml.loader import SafeLoader

_ROOT = os.path.abspath(os.path.dirname(__file__))

# Load parking map
with open(_ROOT + '/parking_map.yml') as f:
    MAP_DATA = yaml.load(f, Loader=SafeLoader)

PARKING_AREAS = MAP_DATA['PARKING_AREAS']
ENTRANCE_AREA = {'min': np.array([5, 70]), 'max': np.array([25, 80])}
WAYPOINTS = MAP_DATA['WAYPOINTS']

class Dataset:

    def __init__(self):
        self.frames = {}
        self.agents = {}
        self.instances = {}
        self.scenes = {}
        self.obstacles = {}
        self.agents_num = 1
        self.agent_pred_dpose = {}

    def load(self, filename):
        # 自己生成第一帧
        # with open(filename + '_frames.json') as f:
        #     self.frames.update(json.load(f))
        # with open(filename + '_agents.json') as f:
        #     self.agents.update(json.load(f))
        # with open(filename + '_instances.json') as f:
        #     self.instances.update(json.load(f))
        with open(filename + '_obstacles.json') as f:
            self.obstacles.update(json.load(f))
        # 加载数据集的内容后，agents信息删除，first_frame和last_frame信息删除
        with open(filename + '_scene.json') as f:
            scene = json.load(f)
            scene['scene_token'] = 'scene_0'
            self.scenes[scene['scene_token']] = scene
        self.init_agents(self.agents_num)
        self.init_frames()
        self.init_scenes()
        self.init_instances()
        frame = self.frames['frame_0']
        for i in range(100):
            self.update_frames()
            for instance_token in frame['instances']:
                instance = self.instances[instance_token]
                rot = np.array([[cos(-instance['heading']), -sin(-instance['heading'])], [sin(-instance['heading']), cos(-instance['heading'])]])

                # update instance pose
                pose = np.array(instance['coords'])
                pose += np.dot(rot, np.array([2*0.04, 0.0]))
                pose = np.append(pose, instance['heading'])
                self.update_instances(instance_token, pose)
            frame = self.frames['frame_' + str(self.frame_id)]


    def init_agents(self, agents_num):
        self.agents_token = []
        for i in range(agents_num):
            agent = {}
            agent['agent_token'] = 'agent_' + str(i)
            agent['scene_token'] = 'scene_0'
            agent['type'] = 'Car'
            agent['size'] = [4.7048, 1.8778]
            agent['first_instance'] = None
            agent['last_instance'] = None
            self.agents[agent['agent_token']] = agent
            self.agents_token.append(agent['agent_token'])

    def init_frames(self):
        frame = {}
        frame['frame_token'] = 'frame_0'
        frame['scene_token'] = 'scene_0'
        frame['timestamp'] = 0.0
        frame['next'] = None
        frame['prev'] = None
        frame['instances'] = []
        self.frames[frame['frame_token']] = frame
        self.frame_id = 0

    def init_scenes(self):
        self.scenes['scene_0']['first_frame'] = 'frame_0'
        self.scenes['scene_0']['last_frame'] = 'frame_0'
        self.scenes['scene_0']['agents'] = self.agents_token

    def init_instances(self):
        self.random_car_init_pose()
        instances_token = []
        for i in range(len(self.agents_token)):
            instance = {}
            instance['frame_token'] = 'frame_0'
            instance['agent_token'] = self.agents_token[i]
            instance['instance_token'] = instance['frame_token'] + '_instance_' + instance['agent_token']
            instance['coords'] = self.car_init_pose[i][:2]
            instance['heading'] = self.car_init_pose[i][2]
            instance['speed'] = 0.0
            instance['acceleration'] = [0.0, 0.0]
            instance['mode'] = 'unclear'
            instance['prev'] = None
            instance['next'] = None
            self.agents[instance['agent_token']]['first_instance'] = instance['instance_token']
            self.agents[instance['agent_token']]['last_instance'] = instance['instance_token']
            instances_token.append(instance['instance_token'])
            self.instances[instance['instance_token']] = instance
        self.frames['frame_0']['instances'] = instances_token
        pass

    def random_car_init_pose(self):
        self.car_init_pose = [[77.70538462, 64.95, 0.],
                              [50.25923077, 64.95, 0],
                              [25.55769231, 64.95, 0],
                              [137.12, 64.95, 3.14],
                              [100.5, 64.95, 3.14],
                              [75.16, 46.82, 0],
                              [9.09, 28.3, 3.14],
                              [137.12, 28.3, 3.14],
                              [85.12, 28.3, 0],
                              [75.16, 9.99, 3.14],
                              [9.09, 9.99, 0],
                              [137.12, 9.99, 0]]

    def _gen_waypoints(self):
        """
        generate waypoints based on yaml
        """
        waypoints = {}
        for name, segment in WAYPOINTS.items():
            bounds = segment['bounds']
            points = np.linspace(bounds[0], bounds[1], num=segment['nums'], endpoint=True)

            waypoints[name] = points

        return waypoints

    def update_frames(self):
        self.frame_id += 1
        new_frame = {}
        new_frame['frame_token'] = 'frame_' + str(self.frame_id)
        new_frame['scene_token'] = 'scene_0'
        new_frame['timestamp'] = self.frame_id * 0.04
        new_frame['next'] = None
        new_frame['prev'] = 'frame_' + str(self.frame_id - 1)
        new_frame['instances'] = []
        self.frames[new_frame['frame_token']] = new_frame
        self.scenes['scene_0']['last_frame'] = new_frame['frame_token']
        self.frames['frame_' + str(self.frame_id - 1)]['next'] = new_frame['frame_token']


    def update_instances(self, instance_token, pose):
        agent_token = self.instances[instance_token]['agent_token']
        new_instance = {}
        new_instance['frame_token'] = 'frame_' + str(self.frame_id)
        new_instance['instance_token'] = new_instance['frame_token'] + '_instance_' + agent_token
        new_instance['agent_token'] = agent_token
        new_instance['coords'] = list(pose[:2])
        new_instance['heading'] = pose[2]
        # new_instance['speed'] = 0.0
        new_instance['acceleration'] = [0.0, 0.0]
        new_instance['mode'] = 'unclear'
        new_instance['prev'] = instance_token
        new_instance['next'] = None
        # 获取上一帧的位置计算速度
        prev_instance = self.instances[instance_token]
        prev_pose = np.array(prev_instance['coords'])
        speed = np.linalg.norm(pose[:2] - prev_pose[:2]) / 0.04
        new_instance['speed'] = float(speed)
        self.instances[instance_token]['next'] = new_instance['instance_token']
        self.instances[new_instance['instance_token']] = new_instance
        self.agents[agent_token]['last_instance'] = new_instance['instance_token']
        self.frames['frame_' + str(self.frame_id)]['instances'].append(new_instance['instance_token'])

    def append_agent_pred_dpose(self, instance_token, pred_dpose):
        self.agent_pred_dpose[self.instances[instance_token]["agent_token"]] = [pred_dpose, instance_token]

    def update(self):
        assert len(self.agent_pred_dpose.keys()) == self.agents_num, "The number of instances in the current frame is not equal to the number of agents."
        frame = self.frames['frame_' + str(self.frame_id)]
        for i in range(1, 11):
            self.update_frames()
            for instance_token in frame['instances']:
                instance = self.instances[instance_token]
                # update instance pose
                current_pose = np.array(self.instances[self.agent_pred_dpose[instance['agent_token']][1]]['coords'])
                current_heading = self.instances[self.agent_pred_dpose[instance['agent_token']][1]]['heading']
                rot = np.array([[cos(-current_heading), -sin(-current_heading)], [sin(-current_heading), cos(-current_heading)]])
                dpose = np.array(self.agent_pred_dpose[instance['agent_token']][0]) / 100
                pose = current_pose + np.dot(rot, dpose[:2]*i)
                pose = np.append(pose, current_heading + dpose[2]*i)
                self.update_instances(instance_token, pose)
                # print(instance['agent_token'], ':',pose)
            frame = self.frames['frame_' + str(self.frame_id)]
        self.agent_pred_dpose = {}

    def get(self, obj_type: str, token: str) -> Dict:
        """
        Get data object as a dictionary

        `obj_type`: string, choose from ['frame', 'agent', 'instance', 'obstacle', 'scene']
        `token`: the corresponding token
        """
        assert obj_type in ['frame', 'agent', 'instance', 'obstacle', 'scene']

        if obj_type == 'frame':
            return self.frames[token]
        elif obj_type == 'agent':
            return self.agents[token]
        elif obj_type == 'instance':
            return self.instances[token]
        elif obj_type == 'obstacle':
            return self.obstacles[token]
        elif obj_type == 'scene':
            return self.scenes[token]
        
    def list_scenes(self) -> List[str]:
        """
        List the tokens of scenes loaded in the current dataset
        """
        return list(self.scenes.keys())

    def get_frame_at_time(self, scene_token: str, timestamp: float, tol=0.039):
        """
        Get the frame object at certain time

        `scene_token`: The scene where the frame comes from
        `timestamp`: time (float) in sec
        `tol`: typically this is the interval between frames
        """
        scene = self.get('scene', scene_token)
        frame_token = scene['first_frame']
        while frame_token:
            frame = self.get('frame', frame_token)
            if abs(frame['timestamp']-timestamp) < tol:
                return frame
            frame_token = frame['next']

        assert frame_token!='', "Didn't find the frame at the specified time. It may exceeds the video length."

    def get_agent_instances(self, agent_token: str) -> List[Dict]:
        """
        Return the list of instance objects for the specific agent

        `agent_token`: Token of the agent
        """
        agent_instances = []
        next_instance = self.agents[agent_token]['first_instance']
        while next_instance:
            inst = self.instances[next_instance]
            agent_instances.append(inst)
            next_instance = inst['next']
        return agent_instances

    def get_agent_future(self, instance_token: str, timesteps: int=5):
        """
        Return a list of future instance objects for the same agent.

        `instance_token`: The token of the current instance
        `timesteps`: (int) Number of steps in the future. 
        """
        return self._get_timeline('instance', 'next', instance_token, timesteps)

    def get_agent_past(self, instance_token: str, timesteps: int=5):
        """
        Return a list of past instance objects for the same agent.

        `instance_token`: The token of the current instance
        `timesteps`: (int) Number of steps in the past. 
        """
        return self._get_timeline('instance', 'prev', instance_token, timesteps)

    def get_future_frames(self, frame_token: str, timesteps: int=5):
        """
        Return a list of future frame objects.

        `frame_token`: The token of the current frame
        `timesteps`: (int) Number of steps in the future. 
        """
        return self._get_timeline('frame', 'next', frame_token, timesteps)
    
    def get_past_frames(self, frame_token: str, timesteps: int=5):
        """
        Return a list of past frame objects.

        `frame_token`: The token of the current frame
        `timesteps`: (int) Number of steps in the past. 
        """
        return self._get_timeline('frame', 'prev', frame_token, timesteps)

    def _get_timeline(self, obj_type, direction, token, timesteps) -> List[Dict]:
        if obj_type == 'frame':
            obj_dict = self.frames
        elif obj_type == 'instance':
            obj_dict = self.instances

        timeline = [obj_dict[token]]
        next_token = obj_dict[token][direction]
        for _ in range(timesteps):
            if not next_token:
                break
            next_obj = obj_dict[next_token]
            timeline.append(next_obj)
            next_token = next_obj[direction]

        if direction == 'prev':
            timeline.reverse()
            
        return timeline

    def signed_speed(self, inst_token: str) -> float:
        """
        Return the speed of the current instance with sign. Positive means it is moving forward, negative measn backward.

        `inst_token`: The token of the current instance
        """
        instance = self.get('instance', inst_token)

        heading_vector = np.array([np.cos(instance['heading']), 
                                   np.sin(instance['heading'])])

        if instance['next']:
            next_inst = self.get('instance', instance['next'])
        else:
            next_inst = instance

        if instance['prev']:
            prev_inst = self.get('instance', instance['prev'])
        else:
            prev_inst = instance
        motion_vector = np.array(next_inst['coords']) - np.array(prev_inst['coords'])

        if heading_vector @ motion_vector > 0:
            return instance['speed']
        else:
            return - instance['speed']

    def get_future_traj(self, inst_token: str, static_thres: float=0.02) -> np.ndarray:
        """
        get the future trajectory of this agent, starting from the current frame
        The static section at the begining and at the end will be truncated

        `static_thres`: the threshold to determine whether it is static. Default is 0.02m/s
        
        Output: T x 4 numpy array. (x, y, heading, speed). T is the time steps
        """
        traj = []

        next_token = inst_token
        while next_token:
            instance = self.get('instance', next_token)
            signed_speed = self.signed_speed(next_token)
            traj.append(np.array([instance['coords'][0], instance['coords'][1], instance['heading'], signed_speed]))

            next_token = instance['next']

        last_idx = len(traj) - 1

        # Find the first non-static index
        idx_start = 0
        while idx_start < last_idx:
            if abs(traj[idx_start][3]) < static_thres:
                idx_start += 1
            else:
                break

        # Find the last non-static index
        idx_end = last_idx
        while idx_end > 0:
            if abs(traj[idx_end][3]) < static_thres:
                idx_end -= 1
            else:
                break

        if idx_end > idx_start:
            return np.array(traj[idx_start:idx_end])
        else:
            # If all indices are static, only return the current time step
            return traj[0].reshape((-1, 4))

    def _inside_parking_area(self, inst_token: str) -> bool:
        """
        check whether the instance is inside the parking area
        """
        instance = self.get('instance', inst_token)
        coords = np.array(instance['coords'])

        for _, area in PARKING_AREAS.items():
            bounds = np.array(area['bounds'])
            bounds_min = np.min(bounds, axis=0)
            bounds_max = np.max(bounds, axis=0)

            if all(coords > bounds_min) and all(coords < bounds_max):
                return True

        return False

    def _ever_inside_parking_area(self, inst_token: str, direction: str):
        """
        check whether the instance is ever inside the parking area

        `direction`: 'prev' - was inside the parking area before, 'next' - will go into the parking area
        """
        result = False
        next_inst_token = inst_token

        while next_inst_token and not result:
            result = self._inside_parking_area(next_inst_token)

            next_inst_token = self.get('instance', next_inst_token)[direction]

        return result

    def _will_leave_through_gate(self, inst_token: str):
        """
        check whether the instance will leave through the gate
        """
        result = False
        next_inst_token = inst_token

        while next_inst_token and not result:
            instance = self.get('instance', next_inst_token)
            coords = np.array(instance['coords'])
            result = all(coords > ENTRANCE_AREA['min']) and all(coords < ENTRANCE_AREA['max']) and instance['heading'] > 0

            next_inst_token = instance['next']

        return result

    def _has_entered_through_gate(self, inst_token: str):
        """
        check whether the instance has entered through entrance
        """
        result = False
        next_inst_token = inst_token

        while next_inst_token and not result:
            instance = self.get('instance', next_inst_token)
            coords = np.array(instance['coords'])
            result = all(coords > ENTRANCE_AREA['min']) and all(coords < ENTRANCE_AREA['max']) and instance['heading'] < 0

            next_inst_token = instance['prev']

        return result
        
    def get_inst_mode(self, inst_token: str, static_thres=0.02) -> str:
        """
        Determine the mode of the vehicle among ["parked", "incoming", "outgoing", 'unclear']. Return the mode as a string and also modify the instance object.
        """
        instance = self.get('instance', inst_token)

        if self._inside_parking_area(inst_token) and instance['speed']<static_thres:
            mode = 'parked'
        elif self._ever_inside_parking_area(inst_token, 'prev'):
            mode = 'outgoing'
        elif self._ever_inside_parking_area(inst_token, 'next'):
            mode = 'incoming'
        elif self._has_entered_through_gate(inst_token):
            mode = 'incoming'
        elif self._will_leave_through_gate(inst_token):
            mode = 'outgoing'
        else:
            mode = 'unclear'

        instance['mode'] = mode

        return mode

    def get_inst_at_location(self, frame_token: str, coords: np.ndarray, exclude_types: Set[str]={'Pedestrian', 'Undefined'}) -> Dict:
        """
        Return the closet instance object (with certain type) at a given location
        `coords`: array-like with two entries [x, y]
        `exclude_types`: the types that we don't want. The reason we exclude types is that the vehicles might have multiple types like 'Car', 'Bus', 'Truck'.
        """
        frame = self.get('frame', frame_token)
        min_dist = np.inf
        min_inst = None
        for inst_token in frame['instances']:
            instance = self.get('instance', inst_token)
            agent = self.get('agent', instance['agent_token'])
            if agent['type'] not in exclude_types:
                x, y = instance['coords']
                dist = (coords[0]-x)**2 + (coords[1]-y)**2
                if dist < min_dist:
                    min_dist = dist
                    min_inst = instance

        return min_inst