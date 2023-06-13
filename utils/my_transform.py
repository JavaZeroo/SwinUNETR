from monai.transforms.transform import Transform
from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from typing import Dict, Hashable, Mapping
from monai.config.type_definitions import NdarrayOrTensor
import torch

# 每一个transform都是要两个类，一个是Transform，一个是MapTransform。Transform是对单个数据进行操作，MapTransform是对多个数据进行操作。MapTransform是放在transforms.Compose(里面的

# 删除channel 可以对照着trainer里面的注释看
class remove_channel(Transform):
    def __init__(self):
        pass
    
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            if len(data.shape) == 4:
                ret = data[0, :, :, :]
            elif len(data.shape) ==  5:
                ret = data[:, 0, :, :, :]
            elif len(data.shape) == 3:
                ret = data
            else:
                raise ValueError(f"Input size: {data.shape} cannot handle")
        return ret
    
class remove_channeld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """
    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, )
        self.adder = remove_channel()
        pass
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d


class change_channel(Transform):
    def __init__(self, back=False):
        self.back = back
        pass
    
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            if not self.back:
                data = data.permute(0, 2, 3, 1)
            else: 
                data = data.permute(0, 3, 1, 2)
        return data
    
class change_channeld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """
    def __init__(self, keys: KeysCollection, back=False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, )
        self.adder = change_channel(back=back)
        pass
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d

# 切片
import random
class z_clip(Transform):
    def __init__(self, num_channel=None, is_3d = False, mid = None, z_list=None, add_shuffled=None):
        self.num_channel = num_channel
        self.is_3d = is_3d
        self.mid = mid
        self.z_list = z_list
        self.add_shuffled = add_shuffled
        if isinstance(z_list, list):
            return
        if num_channel is None:
            raise ValueError("num_channel is None")
        pass
    
    def __call__(self, data):
        if self.z_list is not None:
            start = self.z_list[0]
            end = self.z_list[len(self.z_list)-1]
            if self.is_3d:
                data = data[ :, :, :, start:end]
            else:
                data = data[:, start:end, :, :]
            return data
        
        if self.num_channel == 65:
            return data
        mid = 65 // 2 if self.mid is None else self.mid
        mid = mid + random.randint(-self.add_shuffled, self.add_shuffled) if self.add_shuffled else mid
        start = mid - self.num_channel // 2
        end = mid + self.num_channel // 2
        assert start >= 0 and end <= 65, f"z_clip is not allow for start: {start}, end: {end}"
        if isinstance(data, torch.Tensor):
            if self.is_3d:
                data = data[ :, :, :, start:end]
            else:
                data = data[:, start:end, :, :]
            
        return data
    
class z_clipd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """
    def __init__(self, keys: KeysCollection, num_channel=None, is_3d = False, mid = None, z_list= None, add_shuffled=None) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, )

        self.adder = z_clip(num_channel=num_channel, is_3d=is_3d, mid=mid, z_list=z_list, add_shuffled=add_shuffled)
        pass
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d
    
class printShape(Transform):
    def __init__(self, debug=False):
        self.debug=debug
        pass
    
    def __call__(self, data):
        if self.debug:
            print(data.shape)
        return data
    
class printShaped(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """
    def __init__(self, keys: KeysCollection, debug=False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys)
        self.debug=debug
        pass
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        if not self.debug:
            return dict(data)
        d = dict(data)
        print("==================================")
        for key in self.key_iterator(d):
            print(f"{key}: {d[key].shape}")
        return d
    
class Drop1Layer(Transform):
    def __init__(self):
        pass
    
    def __call__(self, data):
        data_shape = data.shape
        if len(data_shape) == 3:
            _,_,z = data_shape
            ret = data[:, :, :z-1]
        elif len(data_shape) == 4:
            _,_,_,z = data_shape
            ret = data[:, :, :, :z-1]
        else:
            raise Exception("Input demention error")
        return ret
    
class Drop1Layerd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """
    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, )
        self.adder = Drop1Layer()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d
    
class Copy(Transform):
    def __init__(self, num_channel, add_channel=False):
        self.num_channel = num_channel
        self.add_channel = add_channel

    def __call__(self, data):
        if self.add_channel:
            data = data.repeat(1, self.num_channel, 1, 1)  # output = (batch_size=1, num_channel, H, W)
        else:
            data = data.repeat(self.num_channel, 1, 1)  # output = (batch_size=1, num_channel, H, W)
        return data
    
class Copyd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """
    def __init__(self, keys: KeysCollection, num_channel, add_channel=False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, )
        self.adder = Copy(num_channel, add_channel)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d
    

class CustomPad(Transform):
    def __init__(self, target_shape=(96, None, None)):
        self.target_shape = target_shape

    def __call__(self, data):
        if data.shape[0] == self.target_shape[0]:
            return data

        pad_size = max(0, self.target_shape[0] - data.shape[0])
        pad_size_before = pad_size // 2
        pad_size_after = pad_size - pad_size_before

        padded_data = torch.cat((torch.flip(data[:pad_size_before], dims=(0,)), 
                                 data, 
                                 torch.flip(data[-pad_size_after:], dims=(0,))), dim=0)
        return padded_data
    
class CustomPadd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """
    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, )
        self.adder = CustomPad()
        pass
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d