from torch.nn import Module
from geopy.distance import great_circle
import torch
from geopy.point import Point
from geopy.units import radians


class GDLoss(Module):
    def __init__(self):
        super(GDLoss, self).__init__()
        self.pred = None
        self.true = None

    def forward(self, pred, true, device):
        self.pred = pred.clone()
        self.true = true.clone()

        self.pred[:, 0] = self.pred[:, 0] * 25 + 15
        self.pred[:, 1] = self.pred[:, 1] * 360

        self.true[:, 0] = self.true[:, 0] * 25 + 15
        self.true[:, 1] = self.true[:, 1] * 360

        b, m = pred.shape
        sum_gd = torch.zeros(1).to(device=device)
        for i in range(b):
            self._range(i)
            temp_gd = self.measure(self.pred[i, ...], self.true[i, ...])
            sum_gd += temp_gd

        return sum_gd / b

    def _range(self, i):
        if self.pred[i, 0] < - 90:
            self.pred[i, 0] = -90
        elif self.pred[i, 0] > 90:
            self.pred[i, 0] = 90

    def measure(self, a, b):
        a, b = Point(a), Point(b)

        lat1, lng1 = torch.tensor(radians(degrees=a.latitude)), torch.tensor(radians(degrees=a.longitude))
        lat2, lng2 = torch.tensor(radians(degrees=b.latitude)), torch.tensor(radians(degrees=b.longitude))

        sin_lat1, cos_lat1 = torch.sin(lat1), torch.cos(lat1)
        sin_lat2, cos_lat2 = torch.sin(lat2), torch.cos(lat2)

        delta_lng = lng2 - lng1
        cos_delta_lng, sin_delta_lng = torch.cos(delta_lng), torch.sin(delta_lng)

        d = torch.atan2(torch.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                                   (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

        # return self.RADIUS * d
        return d


class CentralAngleLoss(Module):
    def __init__(self):
        super(CentralAngleLoss, self).__init__()
        self.pred = None
        self.true = None

    def forward(self, pred, true):
        m, _ = pred.shape

        self.pred = pred.clone()
        self.true = true.clone()

        self.pred[:, 0] = self.pred[:, 0] * 25 + 15
        self.pred[:, 1] = self.pred[:, 1] * 360

        self.true[:, 0] = self.true[:, 0] * 25 + 15
        self.true[:, 1] = self.true[:, 1] * 360
        # diff_lat = pred[:, 0] - true[:, 0]
        diff_lon = torch.abs(self.pred[:, 1] - self.true[:, 1])

        a = torch.cos(self.true[:, 0]) * torch.sin(diff_lon)
        b = torch.cos(self.pred[:, 0]) * torch.sin(self.true[:, 0]) - torch.sin(self.pred[:, 0]) * torch.cos(self.true[:, 0]) * torch.cos(diff_lon)
        c = torch.sqrt(a ** 2 + b ** 2)
        d = torch.sin(self.pred[:, 0]) * torch.sin(self.true[:, 0]) + torch.cos(self.pred[:, 0]) * torch.cos(self.true[:, 0]) * torch.cos(diff_lon)
        error = torch.atan2(c, d)

        return torch.sum(error, dim=0) / m
