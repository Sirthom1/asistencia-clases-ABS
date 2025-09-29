from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace
from agent import StudentAgent
from shapely.geometry import Point
from geopy.distance import geodesic
import networkx as nx
import geopandas as gpd
import random

DAYS = ["mon", "tue", "wed", "thu", "fri"]

# Helper function to scale municipality coordinates to a 100x100 space
def scale_municipalities(municipalities, scale_to=100):
    """
    Scale municipality coordinates to a continuous 100x100 space.

    Args:
        municipalities: GeoDataFrame with municipality geometries.
        scale_to: Size of the square space to scale to (default 100).

    Returns:
        List of dict: [{"COMUNA": name, "coords": [(x1,y1),...]}]
    """
    minx, miny, maxx, maxy = municipalities.total_bounds
    scaled_municipalities = []

    for idx, municipality in municipalities.iterrows():
        poly_coords = []
        geom = municipality.geometry
        if geom.geom_type == 'Polygon':
            for x, y in geom.exterior.coords:
                x_scaled = (x - minx) / (maxx - minx) * scale_to
                y_scaled = (y - miny) / (maxy - miny) * scale_to
                poly_coords.append((x_scaled, y_scaled))
        scaled_municipalities.append({"municipality": municipality["COMUNA"], "coords": poly_coords})

    return scaled_municipalities

# Helper function to obtain a random point inside a municipality to place a student
def random_point_in_polygon(polygon):
    """
    Generate a random point inside the given polygon.

    Args:
        polygon (shapely.Polygon): The polygon in which the point must be generated.

    Returns:
        tuple: Coordinates (x, y) of the generated point.
    """

    minx, miny, maxx, maxy = polygon.bounds
    while True:
        x, y = random.uniform(minx, maxx), random.uniform(miny, maxy)
        point = Point(x, y)
        if polygon.contains(point):
            return x, y

# Helper function to scale geographic coordinates to a 100x100 space
def scale_coordinates(lon, lat, polygon):
    """
    Scale geographic coordinates to a continuous 100x100 space.

    Args:
        lon (float): Longitude.
        lat (float): Latitude.
        polygon (shapely.Polygon): Reference polygon to compute bounds.

    Returns:
        tuple: Scaled coordinates (x, y).
    """

    minx, miny, maxx, maxy = polygon.bounds

    x_scaled = (lon - minx) / (maxx - minx) * 100  
    y_scaled = (lat - miny) / (maxy - miny) * 100  
    return x_scaled, y_scaled


class ClassAttendanceModel(Model):
    """
    Agent-based model to simulate university class attendance.

    Each agent (student) is assigned a home municipality, a social network,
    and a set of personal and external factors that affect their motivation.
    The model simulates a full week per step, recording attendance, motivation,
    and absence reasons.

    Args:
        num_students (int): Number of student agents.
        min_credits (int): Minimum number of credits per student.
        max_credits (int): Maximum number of credits per student.
        personal_weight (float): Weight of personal motivation.
        social_weight (float): Weight of social influence.
        external_weight (float): Weight of external factors (e.g. weather, distance).
        prob_mandatory (float): Probability that a course is mandatory.
        weather_type (str): Weather configuration ('random' or fixed value).
        network_k (int): Number of initial connections in social network.
        network_p (float): Rewiring probability in the small-world network.
        width (int): Width of the continuous space.
        height (int): Height of the continuous space.
        seed (int): Optional random seed.
    """


    def __init__(self, 
        num_students=10, 
        min_credits=21, 
        max_credits=39, 
        personal_weight=0.5, 
        social_weight=0.3,
        external_weight=0.2,
        prob_mandatory=0.35,
        weather_type="random",
        network_k=4,
        network_p=0.3,
        width=100, 
        height=100, 
        seed=None
    ):
        super().__init__()

        # Parameters
        self.num_students = num_students 
        self.min_credits = min_credits
        self.max_credits = max_credits
        self.prob_mandatory = prob_mandatory
        self.day = "mon" # Start on Monday
        self.weather_type = weather_type
        self.weather = weather_type

        # Normalize weights for motivation
        total = personal_weight + social_weight + external_weight
        self.personal_weight = personal_weight / total
        self.social_weight = social_weight / total
        self.external_weight = external_weight / total

        # Continuous space for agents
        self.space = ContinuousSpace(width, height, torus=False)  

        # Load municipalities and region from shapefiles
        self.municipalities = gpd.read_file("DPA_2023/COMUNAS/COMUNAS_v1.shp")
        self.municipalities = self.municipalities[
            self.municipalities["REGION"] == "Metropolitana de Santiago"
        ].reset_index(drop=True)

        # Load the Metropolitan Region of Santiago to obtain the borders
        self.region = gpd.read_file("DPA_2023/REGIONES/REGIONES_v1.shp")
        self.region = self.region[
            self.region["REGION"] == "Metropolitana de Santiago"
        ][ "geometry"].values[0]
        self.scaled_municipalities = scale_municipalities(self.municipalities)

        # Location of the University of Chile
        self.university = (-70.664314, -33.457596)

        # Create a friendship network using a random graph with a probability p of reconnection and k initial connections
        self.network = nx.connected_watts_strogatz_graph(
            n=self.num_students, k=network_k, p=network_p
        ) # Small-world Network

        # Create students and assign them a random position in a municipality
        for i in range(self.num_students):
            municipality = self.municipalities.sample(1).iloc[0]  
            x, y = random_point_in_polygon(municipality.geometry)

            # Calculate the distance from the municipality to the university 
            # using geodesic to get the km
            distance = geodesic((y, x), self.university[::-1]).km

            x_scaled, y_scaled = scale_coordinates(x, y, self.region)
            student = StudentAgent(self, municipality["COMUNA"], distance)

            # Add the student to the continuous space
            self.space.place_agent(student, (x_scaled, y_scaled))

        # Assign friends to students based on the created graph
        for node in self.network.nodes:
            agent = self.agents[node]
            agent.friends = [self.agents[neighbor] for neighbor in self.network.neighbors(node)]

        # DATA COLLECTION
        self.datacollector = DataCollector(
            model_reporters={"Attendance Rate": self.calculate_attendance_rate,
                             "Attendance": self.calculate_attendance,
                             "Reasons Count": self.reasons_count,
                             "Reasons Effect Count": self.reasons_effect_count, 
                             "Motivation": self.motivation_average,
                             "Weather": lambda m: m.weather,},
            agent_reporters={"Attends Class": lambda a: a.attends_class,
                             "Week Data": lambda a: a.week_data,}
        )
        self.datacollector.collect(self)
    
    def calculate_attendance(self):
        """
        Calculate the total number of students who attended class today.

        Returns:
            int: Number of students who attended.
        """
        attended = sum([agent.attends_class for agent in self.agents])
        return attended

    def calculate_attendance_rate(self):
        """
        Calculate the attendance rate, excluding students with no class.

        Returns:
            float: Attendance rate (0 to 1).
        """
        attended = sum([agent.attends_class for agent in self.agents])
        total = sum([agent.absence_reason != "No tiene clases hoy" for agent in self.agents])
        return attended / total if total > 0 else 0
    
    def reasons_count(self):
        """
        Count the symptomatic reasons students missed class.

        Only considers students who did not attend and have a defined reason.

        Returns:
            dict: {Reason: frequency}
        """
        reasons_count = {}
        # Count the reasons for absence
        for agent in self.agents:
            # Only those who did not attend
            if not agent.attends_class and agent.absence_reason is not None:
                reasons_count[agent.absence_reason] = (
                    reasons_count.get(agent.absence_reason, 0) + 1
                )
        return reasons_count
    
    def reasons_effect_count(self):
        """
        Count the dominant (impact-based) reasons students missed class.

        Only considers students who did not attend and have a defined reason.

        Returns:
            dict: {Reason: frequency}
        """
        reasons_count = {}
        # Count the reasons for absence
        for agent in self.agents:
            # Only those who did not attend and have a defined reason
            if not agent.attends_class and agent.main_factor is not None:
                reasons_count[agent.main_factor] = (
                    reasons_count.get(agent.main_factor, 0) + 1
                )
        return reasons_count
    
    def motivation_average(self):
        """
        Compute the average motivation of all students.

        Returns:
            float: Average motivation (0 to 1).
        """
        motivaciones = [a.motivation for a in self.agents]
        if motivaciones:
            return sum(motivaciones) / len(motivaciones)
        else:
            return 0

    def step(self):
        """
        Execute one simulation step (one full week).

        For each weekday:
        - Assign new weather (if random)
        - Update current day
        - Activate all agents
        - Collect daily metrics
        """

        # Get the new weather
        # If the weather type is "random", a random weather is chosen at each step
        if self.weather_type == "random":
            self.weather = random.choice(
                ["sunny", "rainy", "cold", "hot", "cloudy", "freezing"]
            )

        # The week
        for day in DAYS:
            self.day = day
            # This helps agents activate in a random order and avoids bias
            self.agents.shuffle_do("step")
            self.datacollector.collect(self)
