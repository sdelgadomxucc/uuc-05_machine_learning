import streamlit as st
import calendar
from datetime import date
import pandas as pd
import json
import os

st.title("Calendario – Fechas importantes (Nov 2025 – Ene 2026)")

EVENTS_FILE = "events.json"

# --- Funciones para guardar/cargar en archivo ---

def load_events_from_file():
    if os.path.exists(EVENTS_FILE):
        try:
            with open(EVENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # nos aseguramos que sea diccionario
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}

def save_events_to_file():
    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state["events"], f, ensure_ascii=False, indent=2)

# --- Inicializar diccionario de eventos en la sesión ---
if "events" not in st.session_state:
    # Cargamos de archivo si existe
    st.session_state["events"] = load_events_from_file()

# --- Formulario para agregar/editar eventos ---
st.subheader("Agregar / editar fecha")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Elige una fecha",
        value=date(2025, 11, 1),
        min_value=date(2025, 11, 1),
        max_value=date(2026, 1, 31),
    )

key = selected_date.strftime("%Y-%m-%d")

with col2:
    existing_text = st.session_state["events"].get(key, "")
    activity = st.text_input("Actividad para ese día", value=existing_text)

# Botones: guardar / eliminar la fecha seleccionada
col_guardar, col_eliminar = st.columns(2)

with col_guardar:
    if st.button("Guardar fecha"):
        st.session_state["events"][key] = activity
        save_events_to_file()
        st.success(f"Guardado: {key} → {activity}")

with col_eliminar:
    if st.button("Eliminar fecha seleccionada"):
        if key in st.session_state["events"]:
            st.session_state["events"].pop(key)
            save_events_to_file()
            st.success(f"Se eliminó la fecha {key}.")
        else:
            st.info(f"La fecha {key} no tenía actividad guardada.")

st.markdown("---")

# --- Función para construir un DataFrame tipo calendario ---
def build_month_df(year: int, month: int) -> pd.DataFrame:
    cal = calendar.Calendar(firstweekday=0)  # 0 = lunes
    weeks = cal.monthdayscalendar(year, month)

    data = []
    for week in weeks:
        row = []
        for day in week:
            if day == 0:
                row.append("")  # celda vacía
            else:
                row.append(str(day))
        data.append(row)

    df = pd.DataFrame(
        data,
        columns=["L", "M", "X", "J", "V", "S", "D"]
    )
    return df

def show_month(title: str, year: int, month: int):
    st.markdown(f"#### {title}")
    df = build_month_df(year, month)

    # Estilo: círculo rojo alrededor de las fechas con evento
    def style_events(val):
        if val == "":
            return ""
        if not val.isdigit():
            return ""
        day = int(val)
        k = f"{year}-{month:02d}-{day:02d}"
        if k in st.session_state["events"]:
            return "color: red; border: 2px solid red; border-radius: 50%;"
        return ""

    styler = (
        df.style
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("padding", "0.2rem"),
                    ("font-size", "0.8rem"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                    ("padding", "0.2rem"),
                    ("font-size", "0.85rem"),
                ],
            },
        ])
        .hide(axis="index")        # sin índices
        .applymap(style_events)    # círculo rojo solo a celdas con evento
    )

    st.table(styler)

# --- Mostrar calendarios de noviembre, diciembre y enero ---
st.subheader("Calendario")

col_nov, col_dic, col_ene = st.columns(3)

with col_nov:
    show_month("Noviembre 2025", 2025, 11)

with col_dic:
    show_month("Diciembre 2025", 2025, 12)

with col_ene:
    show_month("Enero 2026", 2026, 1)

st.caption("Las fechas con evento se muestran con un círculo rojo alrededor del número.")

st.markdown("---")

# --- Lista de eventos guardados ---
st.subheader("Fechas marcadas")

if st.session_state["events"]:
    for k, text in sorted(st.session_state["events"].items()):
        st.write(f"📅 **{k}** → {text}")
else:
    st.write("Aún no has marcado ninguna fecha.")


