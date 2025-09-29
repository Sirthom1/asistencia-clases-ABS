from mesa import Agent
import random 

DAYS_OF_WEEK = ["mon", "tue", "wed", "thu", "fri"]
BLOCKS = [
    "08:30-10:00", "10:15-11:45", "12:00-13:30", 
    "14:30-16:00", "16:15-17:45", "18:00-19:30"
]
PATTERN_DAY = {
    3: [["mon"], ["tue"], ["wed"], ["thu"], ["fri"]],
    6: [
        ["mon", "wed"], ["tue", "thu"], 
        ["mon", "fri"],  
        ["wed", "fri"]   
    ],
    12: [
        ["mon", "wed", "fri"], ["tue", "thu", "fri"],
        ["mon", "tue", "thu"]  
    ]
}

class ClassSession:
    """Represents a specific class session defined by day and time block"""
    def __init__(self, day, block):
        self.day = day  
        self.block = block  

class Course:
    """
    Represents a university course.

    Attributes:
        credits (int): Number of credits for the course.
        is_mandatory (bool): Whether the course is mandatory.
        sessions (list): List of ClassSession objects defining the course schedule.
        attendance (float): Attendance percentage to this course.
    """

    def __init__(self, credits, is_mandatory, sessions):
        # self.name_course = name
        # self.professor = professor 
        self.credits = credits
        self.is_mandatory = is_mandatory
        self.sessions = sessions
        self.attendance = 100

    def has_class_on(self, day):
        """Checks if the course has class on the given day."""
        return any(session.day == day for session in self.sessions)

def choose_pattern(credits):
    """Return a valid set of days for a course given its credits."""
    pattern_options = PATTERN_DAY.get(credits, [])
    return random.choice(pattern_options) if pattern_options else None


def find_available_block(schedule, days, use_double_blocks=False):
    """
    Try to find a free block (or double block) across given days.

    Args:
        schedule (dict): Current occupied blocks per day.
        days (list): Days to assign the course.
        use_double_blocks (bool): If True, try to assign two consecutive blocks.

    Returns:
        list of (day, block): Assigned sessions, or None if no valid block found.
    """
    blocks = BLOCKS[:]  # copy
    random.shuffle(blocks)

    for i in range(len(blocks) - (1 if use_double_blocks else 0)):
        block = blocks[i]
        next_block = blocks[i + 1] if use_double_blocks else None

        # Check all days are free for these block(s)
        if all(block not in schedule[d] for d in days) and (
            not use_double_blocks or all(next_block not in schedule[d] for d in days)
        ):
            sessions = []
            for d in days:
                schedule[d].add(block)
                sessions.append(ClassSession(d, block))
                if use_double_blocks:
                    schedule[d].add(next_block)
                    sessions.append(ClassSession(d, next_block))
            return sessions
    return None


def assign_blocks(schedule, days, use_double_blocks=False, same_block=True):
    """
    Assign class sessions to a schedule, either with same block across all days 
    or independent blocks per day.

    Returns:
        list of ClassSession, or None if not possible.
    """
    if same_block:
        return find_available_block(schedule, days, use_double_blocks)

    sessions = []
    for d in days:
        result = find_available_block(schedule, [d], use_double_blocks)
        if result is None:
            return None  # fail if any day can't be scheduled
        sessions.extend(result)
    return sessions


def create_courses(min_credits=21, max_credits=39, prob_mandatory=0.6):
    """
    Generates a randomized list of courses for a student.

    Args:
        min_credits (int): Minimum total credits the student should take.
        max_credits (int): Maximum total credits the student may take.
        prob_mandatory (float): Probability that a course is mandatory.

    Returns:
        list of Course: List of generated courses.
    """
    schedule = {day: set() for day in DAYS_OF_WEEK[:5]}
    total_credits = 0
    course_list = []

    # For fixed cases (e.g., scenario simulations)
    is_fixed = min_credits == max_credits

    while (
        total_credits < min_credits
        or (not is_fixed and total_credits < max_credits and random.random() < 0.7)
    ):
        # Course properties
        is_mandatory = random.random() < prob_mandatory
        credits = random.choice([3, 6, 6, 6, 12])
        days = choose_pattern(credits)
        if not days:
            continue

        # Session assignment strategy
        same_block = random.random() < 0.8
        use_double_blocks = credits >= 6 and random.random() < 0.15

        sessions = assign_blocks(schedule, days, use_double_blocks, same_block)
        if sessions is None:
            continue  # could not place course

        # Create course
        course = Course(credits=credits, is_mandatory=is_mandatory, sessions=sessions)
        course_list.append(course)
        total_credits += credits

    return course_list

class StudentAgent(Agent):
    """
    A student agent with health, motivation, social influence, and class attendance behavior.

    Attributes:
        friends (list): List of friend agents.
        municipality (str): Name of the home municipality.
        distance (float): Distance to university.
        health (float): Health state of the student (0-1).
        motivation (float): Motivation level (0-1).
        courses (list): List of enrolled courses.
        attends_class (bool): Whether the student attends class today.
        absence_reason (str): Reported reason for not attending.
        main_factor (str): Most influential factor for decision.
        week_data (list): Weekly data history.
    """

    
    def __init__(self, model, municipality, distance):
        """
        Initializes a student agent with personal attributes and class schedule.

        Args:
            model (Model): The simulation model to which the agent belongs.
            municipality (str): The municipality where the student lives.
            distance (float): Distance in kilometers from home to the university.
        """
        super().__init__(model)
        # self.school = "Universidad de Chile" 
        # self.year = "" 
        # self.major = "" 
        # self.time_traveling = 0 

        self.friends = [] 
        self.friends_influence = random.uniform(0.0, 1.0) 
        self.new_friends = random.uniform(0.05, 0.35) 
        self.municipality = municipality
        self.distance = distance 
        self.health = random.uniform(0.8, 1.0) 
        self.motivation = random.uniform(0.7, 1.0)   
        self.courses = create_courses(
            self.model.min_credits, self.model.max_credits, self.model.prob_mandatory
        ) 
        self.attends_class = True 
        self.absence_reason = None # Reasons for not attending class
        self.main_factor = None # Most significant reason for not attending class
        self.impact_components = {} # Components impacting motivation (personal, social, external)
        self.day_data = [] # Daily data for the student (used for weekly tracking)
        self.week_data = [] # Weekly tracking data for the student

    @staticmethod
    def weather_impact(weather):
        """Returns the motivation impact of the given weather type."""
        if weather == "sunny":
            return 0.2
        elif weather == "cloudy":
            return 0.1
        elif weather == "rainy":
            return -0.3
        elif weather == "cold":
            return -0.1
        elif weather == "freezing":
            return -0.3
        elif weather == "hot":
            return -0.2
        return 0

    @staticmethod
    def distance_impact(distance):
        """Returns the motivation impact of the given distance (in km)."""
        if distance <= 2:
            return 0.0
        elif distance <= 5:
            return -0.025
        elif distance <= 10:
            return -0.05
        elif distance <= 20:
            return -0.075
        elif distance <= 40:
            return -0.1
        elif distance <= 60:
            return -0.125
        elif distance <= 80:
            return -0.15
        else:
            return -0.2


    def calculate_motivation(self, day):
        """
        Calculates the student's motivation based on personal, social and external factors.
        Also applies a bonus if they have a mandatory course that day.
        """

        # PERSONAL FACTORS
        personal_motivation = self.motivation

        # SOCIAL FACTORS
        friends_motivation = (
            sum(friend.motivation for friend in self.friends) / len(self.friends) 
            if self.friends else 0
        )
        friends_motivation *= self.friends_influence

        # EXTERNAL FACTORS
        weather_mot = self.weather_impact(self.model.weather) 
        distance_mot = self.distance_impact(self.distance)
        external_motivation = weather_mot + distance_mot

        # COMBINATION OF FACTORS FOR FINAL MOTIVATION
        personal = personal_motivation * self.model.personal_weight
        social = friends_motivation * self.model.social_weight
        external = external_motivation * self.model.external_weight

        self.motivation = personal + social + external
        self.motivation = min(1.0, max(0.0, self.motivation))

        # If they have a mandatory course, they have a 25% higher chance of attending
        courses_today = [c for c in self.courses if c.has_class_on(day)]
        has_mandatory_course = any(course.is_mandatory for course in courses_today)

        if has_mandatory_course:
            self.motivation += 0.25
            # Limit motivation between 0.0 and 1.0
            self.motivation = min(1.0, max(0.0, self.motivation))

        # Store the impacts of each factor for analysis
        self.impact_components = {
            "Personal": personal,
            "Social": social,
            "Externo": external
        }

    def decide_attendance(self, day):
        """
        Decides whether the student will attend class, based on motivation, health,
        and whether they have a low-attendance mandatory course.
        """
        # Initial decision based on motivation
        go_class = self.motivation > 0.5

        # Check the classes they have today and if they have a low-attendance mandatory course
        courses_today = [c for c in self.courses if c.has_class_on(day)]
        has_mandatory_course = any(course.is_mandatory for course in courses_today)
        low_attendance = any(course.attendance < 85 for course in courses_today)
        low_health = self.health < 0.35

        # If they have to DO it instead of WANTING to do it
        # If they have low health, they don't attend classes
        if low_health:
            # But if they have a mandatory course with low attendance, 
            # they reconsider if they have to go to class
            if has_mandatory_course and low_attendance:
                go_class = random.random() < 0.4  # 40% chance of attending
                if go_class:
                    self.absence_reason = "Desgaste, pero tiene un ramo obligatorio"
                else:
                    self.absence_reason = "Desgaste"
            else:
                go_class = False
                self.absence_reason = "Desgaste"
        else:
            # If they have a mandatory course with low attendance, they attend class
            if not go_class and has_mandatory_course and low_attendance:
                go_class = True
                self.absence_reason = "Asistencia baja en ramo obligatorio"
            else:
                if go_class:
                    self.absence_reason = "Motivación alta"
                else:
                    self.absence_reason = "Desmotivación"
        
        self.attends_class = go_class

    def update_course_attendance(self, day):
        """Penalizes attendance in mandatory courses if the student misses class."""
        if not self.attends_class:
            courses_today = [c for c in self.courses if c.has_class_on(day)]
            for course in courses_today:
                if course.is_mandatory:
                    # Reduce attendance by 5% for each missed class
                    course.attendance -= 5
                    course.attendance = max(course.attendance, 0)

    def step(self):
        """Performs the agent's daily routine and updates internal state."""

        today = self.model.day
        courses_today = [c for c in self.courses if c.has_class_on(today)]

        if not courses_today:
            self.attends_class = False
            self.absence_reason = "No tiene clases hoy"
        
        else:
            self.calculate_motivation(today)
            self.decide_attendance(today)
            self.update_course_attendance(today)

            if self.attends_class:
                self.main_factor = max(
                    self.impact_components, key=self.impact_components.get
                    )
            else:
                self.main_factor = min(
                    self.impact_components, key=self.impact_components.get
                )

        # Daily data tracking
        self.day_data.append({
            "day": today,
            "attended": self.attends_class,
            "absence_reason": self.absence_reason,
            "main_factor": self.main_factor,
            "motivation": self.motivation,
            "health": self.health,
            "courses": self.courses,
        })

        # Students interact with their friends
        self.interact()

        if self.attends_class:
            # If attending class while very fatigued, lose more health
            if self.health < 0.35:
                self.health -= random.uniform(0.05, 0.25)
            else:
                self.health -= random.uniform(0.01, 0.1)
        else:
            # If not attending class, recover some health
            self.health += random.uniform(0.05, 0.15)
        
        if self.model.day == "fri":
            # At the end of the week, recover a bit more health
            self.health += random.uniform(0.1, 0.5)

            # Weekly data tracking
            self.week_data.append({
            "week": self.model.steps,
            "days": self.day_data
            })
            self.day_data = [] 

        self.health = max(0.0, min(1.0, self.health))

    def interact(self):
        """Chance to make new friends depending on class attendance."""
        if not self.attends_class:
            probability = self.new_friends - 0.05 
        else:
            probability = self.new_friends 

        # A chance to make a new friend
        if random.random() < probability:  
            potential_new_friends = [
                agent for agent in self.model.agents if agent != self and agent not in self.friends
            ]
            if potential_new_friends:
                new_friend = random.choice(potential_new_friends)
                self.friends.append(new_friend)
                new_friend.friends.append(self)  