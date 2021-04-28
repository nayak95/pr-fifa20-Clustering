import pandas as pd

class S_Sets:
    data_paths = [
        'data/s_sets/s1.txt',
        'data/s_sets/s2.txt',
        'data/s_sets/s3.txt',
        'data/s_sets/s4.txt'
    ]

    label_paths = [
        'data/s_sets/s1-label.pa',
        'data/s_sets/s2-label.pa',
        'data/s_sets/s3-label.pa',
        'data/s_sets/s4-label.pa'
    ]

    @staticmethod
    def get_data(i):
        data = pd.read_csv(S_Sets.data_paths[i-1], sep="\s+", header=None)
        data.columns = ["x", "y"]
        return data

    @staticmethod
    def get_labels(i):
        read_labels = pd.read_csv(S_Sets.label_paths[i-1], header=None).to_numpy()
        true_labels = [a[0] for a in read_labels]
        return true_labels

class Spiral:
    data_path = 'data/spiral.txt'

    @staticmethod
    def get_data():
        all_data = pd.read_csv(Spiral.data_path, sep="\s+", header=None)
        data = all_data.iloc[:, 0:2]
        data.columns = ["x", "y"]
        return data

    @staticmethod
    def get_labels():
        all_data = pd.read_csv(Spiral.data_path, sep="\s+", header=None)
        labels = all_data.iloc[:, 2:].to_numpy()
        true_labels = [a[0] for a in labels]
        return true_labels


class FIFA20:
    data_path = 'data/fifa20.csv'

    skills = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing',
        'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',
        'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',
        'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
        'LongShots', 'Aggression', 'Interceptions', 'Positioning',
        'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle',
        'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking',
        'GKPositioning', 'GKReflexes']

    @staticmethod
    def get_data():
        all_data = pd.read_csv(FIFA20.data_path)
        # print(len(all_data.columns), all_data.columns)
        skills_data = all_data[FIFA20.skills]
        skills_data.dropna(inplace=True)
        # print(len(skills_data.columns), skills_data)
        return skills_data