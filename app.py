import solara
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from model import ClassAttendanceModel
from mesa.visualization import SolaraViz
from mesa.visualization.utils import update_counter
from collections import defaultdict

# ----------------------------
# Helper functions
# ----------------------------

def get_model_data(model):
    """
    Centralize model data extraction to avoid code repetition.

    Args:
        model: Instance of ClassAttendanceModel.

    Returns:
        agent_data (DataFrame), model_data (DataFrame), gdf (GeoDataFrame)
    """
    agent_data = model.datacollector.get_agent_vars_dataframe().reset_index()
    model_data = model.datacollector.get_model_vars_dataframe().reset_index()
    gdf = model.scaled_municipalities
    return agent_data, model_data, gdf

def get_top_reason(reasons_series):
    """
    Given a series of dictionaries with reasons and their counts,
    return the reason with the highest total count.
    Args:
        reasons_series: pd.Series of dicts.
    Returns:
        str: reason with the highest count or "Ninguna" if none.
    """
    all_reasons = defaultdict(int)
    for day_reasons in reasons_series:
        if isinstance(day_reasons, dict):
            for k, v in day_reasons.items():
                all_reasons[k] += v
    return max(all_reasons.items(), key=lambda x: x[1])[0] if all_reasons else "Ninguna"


def compute_agent_stats(agent, metric, mode):
    """
    Calculate the attendance or motivation average for an agent,
    depending on the metric and mode.

    Args:
        agent: instance of StudentAgent.
        metric: str, "Asistencia" or "Motivación".
        mode: str, "Histórico" or "Semanal".

    Returns:
        value: calculated value (float between 0 and 1).
    """
    total_attendance = 0
    days_with_classes = 0
    motivation_total = 0
    motivation_days = 0

    week_data = agent.week_data if mode == "Histórico" else agent.week_data[-1:]

    for week in week_data:
        for day in week["days"]:
            if day.get("reason_symptom") != "No tiene clases hoy":
                days_with_classes += 1
                if day.get("attended"):
                    total_attendance += 1
            if "motivation" in day:
                motivation_total += day["motivation"]
                motivation_days += 1

    if metric == "Asistencia":
        return total_attendance / days_with_classes if days_with_classes > 0 else 0
    else: 
        return motivation_total / motivation_days if motivation_days > 0 else 0


# ----------------------------
# Model parameters
# ----------------------------

model_params = {
    "num_students": {
        "type": "SliderInt",
        "value": 10,
        "label": "Número de agentes:",
        "min": 5,
        "max": 100,
        "step": 1,
    },
    "min_credits": {
        "type": "SliderInt",
        "value": 21,
        "label": "Mínimo de créditos:",
        "min": 0,
        "max": 60,
        "step": 1,
    },
    "max_credits": {
        "type": "SliderInt",
        "value": 39,
        "label": "Máximo de créditos:",
        "min": 0,
        "max": 60,
        "step": 1,
    },
    "personal_weight": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "Peso de motivación personal:",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "social_weight": {
        "type": "SliderFloat",
        "value": 0.3,
        "label": "Peso de motivación social:",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "external_weight": {
        "type": "SliderFloat",
        "value": 0.2,
        "label": "Peso de motivación externa:",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "prob_mandatory": {
        "type": "SliderFloat",
        "value": 0.6,
        "label": "Prob. alguna clase obligatoria:",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "weather_type": {
        "type": "Select",
        "value": "random",
        "label": "Tipo de clima:",
        "values": ["random", "sunny", "rainy", "cold", "hot", "cloudy", "freezing"],
    },
    "network_k": {
        "type": "SliderInt",
        "value": 4,
        "label": "Número de amigos(k):",
        "min": 1,
        "max": 10,
        "step": 1,
    },
    "network_p": {
        "type": "SliderFloat",
        "value": 0.3,
        "label": "Prob. de amistad(p):",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "width": 100,
    "height": 100,
}

attendance_model = ClassAttendanceModel(
                        num_students=10, min_credits=21, max_credits=39,
                        personal_weight=0.5, social_weight=0.3, external_weight=0.2,
                        prob_mandatory=0.6, weather_type="random", network_k=4,
                        network_p=0.3, width=100, height=100
                    )

# ----------------------------
# Reactive UI variables
# ----------------------------

metric = solara.reactive("Asistencia")
metrics = ["Asistencia", "Motivación"]

mode = solara.reactive("Histórico")
modes = ["Histórico", "Semanal"]

show_edges = solara.reactive("Activar")
show_edges_values = ["Activar", "Desactivar"]

reason_mode = solara.reactive("Síntomas")
reason_modes = ["Síntomas", "Dominante"]

show_cards = solara.reactive(False)


# ----------------------------
# Components
# ----------------------------

@solara.component
def InfoCards(model):
    """
    Component that displays informational cards with key model data:
    current week, weather, total students, average daily and weekly attendance,
    average motivation, and the most common reason for absence.

    Args:
        model: Instance of ClassAttendanceModel.
    """
    update_counter.get() 
    total_students = len(model.agents)
    current_step = model.steps
    _, data, _ = get_model_data(model)

    last_attendance = data["Attendance"].iloc[-5:] if not data[1:].empty else pd.Series([])
    last_attendance_mean = last_attendance.mean()  if not last_attendance.empty else 0
    attendance_pct = (last_attendance.sum() / (total_students * 5) * 100) if ((total_students > 0) and (not data[1:].empty)) else 0
    
    last_motivation = data["Motivation"].iloc[-5:] if not data.empty else pd.Series([])
    avg_motivation = last_motivation.mean() if not last_motivation.empty else 0

    reasons = data["Reasons Count"].iloc[-5:] if not data.empty else pd.Series([])
    top_reason = get_top_reason(reasons) if not reasons.empty else "Ninguna"

    with solara.ColumnsResponsive(12, large=1): 
        with solara.Card():
            solara.Markdown(f"#### 📆 Semana Actual: **{current_step}**")

        with solara.Card():
            solara.Markdown(f"#### 🌤️ Clima de la semana: **{model.weather.capitalize()}**")

        with solara.Card():
            solara.Markdown(f"#### 👥 Total Estudiantes: **{total_students}**")

        with solara.Card():
            solara.Markdown(f"#### ✅ Asistencia promedio diaria: **{last_attendance_mean:.2f}**")

        with solara.Card():
            solara.Markdown(f"#### 📊 % Asistencia semanal: **{attendance_pct:.1f}%**")

        with solara.Card():
            solara.Markdown(f"#### 🧠 Motivación promedio semanal: **{avg_motivation:.1f}**")

        with solara.Card():
            solara.Markdown(f"#### 🚫 Razón más común de inasistencia: **{top_reason}**")


@solara.component
def DashboardControls():
    """
    Component with controls to select the metric to display (Attendance or Motivation),
    the visualization mode (Historical or Weekly), and toggle friendship network visibility on the map.
    """
    update_counter.get()
    return solara.Card(title="Controles de visualización", children=[
        solara.Row([
            solara.Select(label="Métrica", value=metric, values=metrics),
            solara.Select(label="Modo de visualización", value=mode, values=modes),
            solara.Select(label="Mostrar amistades", value=show_edges, values=show_edges_values),
        ])
    ])


@solara.component
def SantiagoMapWithAgents(model):
    """
    Displays an interactive map of Santiago with scaled municipalities and student locations,
    colored according to the selected metric (Attendance or Motivation). Optionally draws friendship edges.

    Args:
        model: Instance of ClassAttendanceModel.
    """
    update_counter.get()
    _, _, gdf = get_model_data(model)
    fig = go.Figure()

    for comuna in gdf:
        comuna_name = comuna["municipality"]
        xs, ys = zip(*comuna["coords"])
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            fill="toself",
            mode="lines",
            line=dict(color="black", width=0.5),
            name=comuna_name,
            hoverinfo="text",
            text=f"Comuna: {comuna_name}",
            showlegend=False,
            opacity=0.3
        ))

    valores, x_pos, y_pos, hover_texts = [], [], [], []

    for agent in model.agents:
        val = compute_agent_stats(agent, metric.value, mode.value)
        valores.append(val)
        x_pos.append(agent.pos[0])
        y_pos.append(agent.pos[1])
        hover_texts.append(
            f"Estudiante: {agent.unique_id}<br>"
            f"Comuna: {agent.municipality}<br>"
            f"{metric.value}: {val:.2f}"
        )

    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode="markers",
        marker=dict(
            color=valores,
            colorscale="RdYlGn" if metric.value == "Asistencia" else "plasma",
            cmin=0, cmax=1,
            size=8,
            colorbar=dict(title=metric.value),
        ),
        text=hover_texts,
        hoverinfo="text",
        name="Estudiantes"
    ))

    if show_edges.value == "Activar":
        for agent in model.agents:
            for friend in agent.friends:
                fig.add_trace(go.Scatter(
                    x=[agent.pos[0], friend.pos[0]],
                    y=[agent.pos[1], friend.pos[1]],
                    mode="lines",
                    line=dict(color="gray", width=0.5),
                    hoverinfo="skip",
                    showlegend=False,
                    opacity=0.4
                ))

    fig.update_layout(
        title=f"Mapa de estudiantes por {metric.value.lower()} ({mode.value.lower()})",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        showlegend=False
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(
        height=600, 
        margin=dict(l=60, r=30, t=60, b=40)
    )

    return solara.FigurePlotly(fig)

@solara.component
def AttendanceOrMotivationByMunicipalityBar(model):
    """
    Displays horizontal bar charts representing average attendance or motivation per municipality,
    depending on the selected metric and mode (Historical or Weekly).

    Args:
        model: Instance of ClassAttendanceModel.
    """
    update_counter.get()
    
    df, _, _ = get_model_data(model)
    num_agents = len(model.agents)
    df = df.tail(num_agents)

    df = df.merge(pd.DataFrame([
        {"AgentID": agent.unique_id, "Comuna": agent.municipality}
        for agent in model.agents
    ]), on="AgentID")

    registros = []

    for agent in model.agents:
        val = compute_agent_stats(agent, metric.value, mode.value)
        registros.append({"Comuna": agent.municipality, "Valor": val})

    df_registros = pd.DataFrame(registros)

    if df_registros.empty:
        promedio = pd.Series([0], index=["Sin datos"])
    else:
        promedio = df_registros.groupby("Comuna")["Valor"].mean().sort_values(ascending=True)

    promedio_df = promedio.reset_index()
    promedio_df.columns = ["Comuna", "Valor"]

    label = "Motivación promedio" if metric.value == "Motivación" else "% Asistencia promedio"
    color_scale = "plasma" if metric.value == "Motivación" else "RdYlGn"

    fig = px.bar(
        promedio_df,
        x="Valor",
        y="Comuna",
        orientation="h",
        color="Valor",
        range_color=[0, 1],
        color_continuous_scale=color_scale,
        labels={"Valor": label},
        hover_data={"Valor": False},  
        title=f"{label} por comuna ({mode.value.lower()})"
    )

    fig.update_traces(
        hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.2f}}<extra></extra>"
    )

    fig.update_layout(
        height=600,  
        margin=dict(l=60, r=30, t=60, b=40)
    )

    fig.update_layout(autosize=True)

    return solara.FigurePlotly(fig)

@solara.component
def AttendanceTrend(model):
    """
    Shows a stacked bar chart of the weekly evolution of total attendance and absences.

    Args:
        model: Instance of ClassAttendanceModel.
    """ 
    update_counter.get()

    data, _, _ = get_model_data(model)
    if data["Week Data"][0] == []:
        return solara.Text("Sin datos aún")

    num_agents = len(model.agents)  
    last_rows = data.tail(num_agents)

    semanal = {}

    for _, row in last_rows.iterrows():
        week_data = row["Week Data"]
        for semana in week_data:
            week = semana.get("week")
            if week not in semanal:
                semanal[week] = {"Asistencias": 0, "Inasistencias": 0}

            for dia in semana.get("days", []):
                if dia.get("attended", False):
                    semanal[week]["Asistencias"] += 1
                elif dia.get("absence_reason", "") != "No tiene clases hoy":
                    semanal[week]["Inasistencias"] += 1

    df = pd.DataFrame.from_dict(semanal, orient="index").sort_index()
    df.index.name = "Semana"

    fig = go.Figure()

    colores = {
        "Asistencias": "green",
        "Inasistencias": "red",
    }

    for categoria in ["Asistencias", "Inasistencias"]:
        fig.add_trace(go.Bar(
            x=df.index.astype(str),
            y=df[categoria],
            name=categoria,
            marker_color=colores[categoria],
            hovertemplate=f"%{{y}} {categoria.lower()}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title="Evolución de la Asistencia Semanal",
        xaxis_title="Semana",
        yaxis_title="Cantidad",
        height=500,
        legend_title="Tipo de registro",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return solara.FigurePlotly(fig)

@solara.component
def Reasons(model):
    """
    Presents a stacked bar chart of the weekly evolution of absence reasons,
    allowing toggling between specific symptoms or dominant motivational factors.

    Args:
        model: Instance of ClassAttendanceModel.
    """
    update_counter.get()

    mode = reason_mode.value

    _, data, _ = get_model_data(model)

    if mode == "Síntomas":
        reasons_data = data["Reasons Count"][1:] 
    else:
        reasons_data = data["Reasons Effect Count"][1:]

    filtered = reasons_data[
        reasons_data.apply(lambda d: isinstance(d, dict) and any(v > 0 for v in d.values()))
    ]

    if filtered.empty:
        return solara.Text("Sin datos aún")
    else:
        expanded = filtered.apply(pd.Series).fillna(0)

        step_interval = 5
        grouped = []
        for i in range(0, len(expanded), step_interval):
            chunk = expanded.iloc[i:i + step_interval]
            grouped.append(chunk.sum())

        df_weekly = pd.DataFrame(grouped, index=range(1, len(grouped) + 1))
        df_weekly.index.name = "Semana"

        reason_totals = df_weekly.sum().sort_values(ascending=False)
        ordered_columns = reason_totals.index.tolist()
        df_weekly = df_weekly[ordered_columns]

        fig = go.Figure()
        for col in df_weekly.columns:
            fig.add_trace(go.Bar(
                x=df_weekly.index.astype(str),
                y=df_weekly[col],
                name=col,
                hovertemplate='%{y} estudiantes<extra>%{fullData.name}</extra>'
            ))
        
        if mode == "Síntomas":
            chart_title = 'Evolución Semanal de Razones de Inasistencia (Síntomas)'
            legend_title = 'Síntomas'
        else: 
            chart_title = 'Evolución Semanal de Factores Motivacionales Dominantes'
            legend_title = 'Factores Dominantes'

        fig.update_layout(
            barmode='stack',
            title=chart_title,
            xaxis_title='Semana',
            yaxis_title='Número de estudiantes',
            height=500,
            margin=dict(l=40, r=40, t=60, b=40),
            legend_title_text=legend_title
        )

    return solara.Column([
            solara.Select(label="Razón a visualizar", value=reason_mode, values=reason_modes),
            solara.FigurePlotly(fig)
        ])

@solara.component
def Motivation(model):
    """
    Displays the weekly average evolution of student motivation and attendance in a line chart.

    Args:
        model: Instance of ClassAttendanceModel.
    """
    update_counter.get()

    _, data, _ = get_model_data(model)
    motivation_data = data["Motivation"][1:]
    attendance_data = data["Attendance Rate"][1:]

    if motivation_data.empty or attendance_data.empty:
        return solara.Text("Sin datos aún")

    step_interval = 5
    weekly_motivation = []
    weekly_attendance = []

    for i in range(0, len(motivation_data), step_interval):
        motiv_chunk = motivation_data.iloc[i:i + step_interval]
        att_chunk = attendance_data.iloc[i:i + step_interval]

        weekly_motivation.append(motiv_chunk.mean())
        weekly_attendance.append(att_chunk.mean())

    semanas = list(range(1, len(weekly_motivation) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=semanas,
        y=weekly_motivation,
        mode="lines+markers",
        line=dict(color="orange", width=2),
        marker=dict(size=6),
        name="Motivación promedio",
        hovertemplate="Semana %{x}<br>Motivación: %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=semanas,
        y=weekly_attendance,
        mode="lines+markers",
        line=dict(color="green", width=2, dash="dot"),
        marker=dict(size=6),
        name="Asistencia promedio",
        hovertemplate="Semana %{x}<br>Asistencia: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="Evolución Semanal de Motivación y Asistencia",
        xaxis_title="Semana",
        yaxis_title="Promedio",
        yaxis=dict(range=[0, 1]),
        height=500,
        legend_title="Métrica",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return solara.FigurePlotly(fig)

@solara.component
def HeatmapAssistance(model):
    """
    Generates a heatmap over the map of Santiago municipalities, showing the average
    attendance rate historically or for the last week, coloring municipalities accordingly.

    Args:
        model: Instance of ClassAttendanceModel.
    """
    update_counter.get()

    df, _, gdf = get_model_data(model)
    last_rows = df.tail(len(model.agents))

    last_rows = last_rows.merge(pd.DataFrame([
        {"AgentID": agent.unique_id, "Comuna": agent.municipality}
        for agent in model.agents
    ]), on="AgentID")

    asistencia_por_comuna = defaultdict(lambda: {"asistencias": 0, "dias_con_clase": 0})

    for _, row in last_rows.iterrows():
        comuna = row["Comuna"]
        semanas = row["Week Data"]

        for semana in semanas:
            for dia in semana.get("days", []):
                if dia.get("absence_reason") != "No tiene clases hoy":
                    asistencia_por_comuna[comuna]["dias_con_clase"] += 1
                    if dia.get("attended", False):
                        asistencia_por_comuna[comuna]["asistencias"] += 1

    porcentajes = {
        comuna: datos["asistencias"] / datos["dias_con_clase"]
        if datos["dias_con_clase"] > 0 else 0
        for comuna, datos in asistencia_por_comuna.items()
    }

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = matplotlib.colormaps.get_cmap("Greens")

    fig = go.Figure()

    for comuna in gdf:
        comuna_name = comuna["municipality"]
        porcentaje = porcentajes.get(comuna_name, 0)
        color_rgba = cmap(norm(porcentaje))
        color_hex = mcolors.to_hex(color_rgba)
        xs, ys = zip(*comuna["coords"])

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            fill="toself",
            mode="lines",
            line=dict(color="black", width=0.5),
            fillcolor=color_hex,
            name=comuna_name,
            showlegend=False,
            text=f"{comuna_name}<br>Asistencia: {porcentaje*100:.1f}%",
            hoverinfo="text",
            opacity=0.8
        ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale='Greens',
            cmin=0,
            cmax=1,
            color=0.5,
            showscale=True,
            colorbar=dict(title="Asistencia", tickformat=".0%")
        ),
        hoverinfo='none',
        showlegend=False
    ))

    fig.update_layout(
        title=f"Mapa de calor de asistencia por comuna (Histórico)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return solara.FigurePlotly(fig)

@solara.component
def SummaryWeekly(model):
    """
    Table summarizing weekly statistics such as average attendance, average motivation,
    motivation variation, weather, and most common reason for absence.

    Args:
        model: Instance of ClassAttendanceModel.
    """
    update_counter.get()
    _, data_model, _ = get_model_data(model)
    
    semanas = []
    dias_por_semana = 5

    for semana_idx in range(1, len(data_model), dias_por_semana):  
        semana = data_model.iloc[semana_idx:semana_idx + dias_por_semana]

        asistencia_prom = semana["Attendance Rate"].mean()
        motivacion_prom = semana["Motivation"].mean()

        motivacion_var = semana["Motivation"].max() - semana["Motivation"].min()

        clima_semana = semana["Weather"].iloc[0] if "Weather" in semana else "Desconocido"

        razones_sumadas = pd.Series(dtype=int)
        for r in semana["Reasons Count"]:
            if isinstance(r, dict):
                razones_sumadas = razones_sumadas.add(pd.Series(r), fill_value=0)

        if not razones_sumadas.empty:
            razon_mas_comun = razones_sumadas.idxmax()
        else:
            razon_mas_comun = "Sin datos"

        fila = {
            "Semana": semana_idx // dias_por_semana,
            "Clima": clima_semana,
            "Asistencia promedio (%)": round(asistencia_prom * 100, 1),
            "Motivación promedio": round(motivacion_prom, 2),
            "Variación motivacional": round(motivacion_var, 2),
            "Razón más común": razon_mas_comun
        }

        semanas.append(fila)

    resumen_df = pd.DataFrame(semanas)
    return solara.DataFrame(resumen_df, items_per_page=10)


@solara.component
def SummaryWeeklyAgent(model):
    """
    Weekly performance summary table for each student, including municipality,
    attendance percentage, average motivation, and most common reason for absence.

    Args:
        model: Instance of ClassAttendanceModel.
    """
    update_counter.get()
    df, _, _ = get_model_data(model)

    num_agents = len(model.agents)
    last_rows = df.tail(num_agents)

    resumen = []

    for _, row in last_rows.iterrows():
        agent_id = row["AgentID"]
        comuna = next((a.municipality for a in model.agents if a.unique_id == agent_id), "Desconocida")
        week_data = row["Week Data"]
        if not week_data:
            continue
        dias = week_data[-1].get("days", [])

        total_dias = sum(1 for d in dias if d.get("absence_reason") != "No tiene clases hoy")
        asistencias = sum(1 for d in dias if d.get("attended", False))
        motivation = [d.get("motivation", 0) for d in dias if "motivation" in d]
        motivation_promedio = round(sum(motivation) / len(motivation), 2) if motivation else 0

        razones = [d.get("absence_reason") for d in dias if d.get("absence_reason") not in [None, "", "No tiene clases hoy"]]
        if razones:
            top_reason = max(set(razones), key=razones.count)
        else:
            top_reason = "Sin registro"

        resumen.append({
            "Estudiante": agent_id,
            "Comuna": comuna,
            "Asistencia (%)": round((asistencias / total_dias) * 100, 2) if total_dias > 0 else 0,
            "Motivación promedio": motivation_promedio,
            "Razón más común": top_reason
        })

    resumen_df = pd.DataFrame(resumen)
    return solara.DataFrame(resumen_df, items_per_page=10)

@solara.component
def DailyDetail(model):
    """
    Detailed daily records table for each student: week, day, attendance, reason for absence,
    motivation, and municipality.

    Args:
        model: Instance of ClassAttendanceModel.
    """
    update_counter.get()
    df, _, _ = get_model_data(model)
    num_agents = len(model.agents)
    last_rows = df.tail(num_agents)

    detalles = []

    dias_traducidos = {
        "mon": "Lunes",
        "tue": "Martes",
        "wed": "Miércoles",
        "thu": "Jueves",
        "fri": "Viernes",
    }

    for _, row in last_rows.iterrows():
        agent_id = row["AgentID"]
        comuna = next((a.municipality for a in model.agents if a.unique_id == agent_id), "Desconocida")
        for semana in row["Week Data"]:
            week = semana.get("week")
            for dia in semana.get("days", []):
                detalles.append({
                    "Estudiante": agent_id,
                    "Semana": week,
                    "Día": dias_traducidos.get(dia.get("day", ""), dia.get("day", "")),
                    "Asistió": "Sí" if dia.get("attended") else "No",
                    "Razón": dia.get("absence_reason", ""),
                    "Motivación": round(dia.get("motivation"),2),
                    "Comuna": comuna,
                })

    detalles_df = pd.DataFrame(detalles)
    return solara.DataFrame(detalles_df, items_per_page=10)

# Componente principal de la aplicación
@solara.component
def App(model):
    """
    Main application component managing tabs ("Map", "Attendance Evolution", "Data") and
    coordinating the display of different components based on the selected tab.

    Args:
        model: Instance of ClassAttendanceModel.
    """
    tab = solara.use_reactive(0)
    with solara.Card():
        with solara.ColumnsResponsive([1]):
            solara.ToggleButtonsSingle(
                value=tab.value,
                values=["Mapa", "Evolución Asistencia", "Datos"],
                on_value=lambda v: tab.set(v),
            )


        if tab.value == "Mapa":
            with solara.Card():
                InfoCards(model)
            DashboardControls()
            with solara.ColumnsResponsive(12, large=[6, 6]):
                with solara.Card():
                    SantiagoMapWithAgents(model)
                with solara.Card():
                    AttendanceOrMotivationByMunicipalityBar(model)

        elif tab.value == "Evolución Asistencia":
            label = "Ocultar tarjetas de resumen" if show_cards.value else "Ver tarjetas de resumen"
            with solara.Card():
                solara.Button(label, on_click=lambda: show_cards.set(not show_cards.value))
                if show_cards.value:
                    with solara.Card():
                        InfoCards(model)
            with solara.Card():
                with solara.ColumnsResponsive(12, large=[6, 6]):
                    with solara.Card():
                        AttendanceTrend(model)
                    with solara.Card():
                        Motivation(model)
                    with solara.Card():
                        Reasons(model)
                    with solara.Card():
                        HeatmapAssistance(model)
        elif tab.value == "Datos":
            with solara.Card():
                SummaryWeekly(model)
            with solara.Card():
                SummaryWeeklyAgent(model)
            with solara.Card():
                DailyDetail(model)

# Poner la aplicación en marcha con controles
page = SolaraViz(
    attendance_model,
    components=[App],  # Componente con las pestañas
    model_params=model_params,
    name="Class Attendance Model",
)
