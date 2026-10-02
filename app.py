import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import io

st.set_page_config(page_title="Optimus Engine Core", page_icon="🦿", layout="wide")
st.title("🦿 Optimus Structural Kinematics & AI Synthesis Engine")

st.sidebar.header("🌍 Simulation Constraints")
gravity = st.sidebar.slider("Gravitational Forces (m/s²)", 1.0, 25.0, 9.81)
terrain_friction = st.sidebar.slider("Surface Friction Coefficient", 0.1, 1.0, 0.6)
ankle_mass = st.sidebar.slider("Ankle Segment (kg)", 1.0, 30.0, 8.5)
knee_mass = st.sidebar.slider("Knee Segment (kg)", 1.0, 50.0, 16.0)

col1, col2 = st.columns()
dt, total_steps = 0.01, 500
time_axis = np.arange(total_steps) * dt

np.random.seed(101)
terrain = np.sin(time_axis * 2) * 0.15
terrain[200:] += 0.4 

ankle_angles, knee_angles = np.zeros(total_steps), np.zeros(total_steps)
a_angle, a_vel, k_angle, k_vel = 0.0, 0.0, 0.0, 0.0

for t in range(1, total_steps):
    a_err = terrain[t] - a_angle
    a_torque = np.clip((a_err * 220.0) - (a_vel * 20.0), -250.0, 250.0)
    a_accel = (a_torque - (ankle_mass * gravity * np.sin(a_angle) * terrain_friction)) / ankle_mass
    a_vel += a_accel * dt
    a_angle += a_vel * dt
    ankle_angles[t] = a_angle

    k_err = -a_angle - k_angle
    k_torque = np.clip((k_err * 280.0) - (k_vel * 25.0), -400.0, 400.0)
    k_accel = (k_torque - (knee_mass * gravity * np.sin(k_angle))) / knee_mass
    k_vel += k_accel * dt
    k_angle += k_vel * dt
    knee_angles[t] = k_angle

with col1:
    st.subheader("📈 Kinematic Deflection Analysis")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, terrain, 'r:', label="Surface Impulse")
    ax.plot(time_axis, ankle_angles, 'b-', label="Ankle Shift")
    ax.plot(time_axis, knee_angles, 'g--', label="Knee Compensation")
    ax.set_facecolor("#111625")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(facecolor="#111625", edgecolor="none", labelcolor="white")
    st.pyplot(fig)

with col2:
    st.subheader("⚡ Databank Synthesizer")
    batch_size = st.number_input("Target Scenario Variations", 100, 10000, 1000, 100)

    if st.button("🚀 Process & Generate AI Tensors"):
        status = st.empty()
        bar = st.progress(0.0)
        records = []

        for sim_id in range(int(batch_size)):
            g_var = np.random.uniform(3.7, 20.0)
            f_var = np.random.uniform(0.2, 0.9)
            target_impulse = np.random.uniform(-0.6, 0.6)
            s_a_angle, s_a_vel, s_k_angle, s_k_vel = 0.0, 0.0, 0.0, 0.0

            for step in range(300):
                ea = target_impulse - s_a_angle
                ta = np.clip((ea * 220.0) - (s_a_vel * 20.0), -250.0, 250.0)
                s_a_vel += ((ta - (ankle_mass * g_var * np.sin(s_a_angle) * f_var)) / ankle_mass) * dt
                s_a_angle += s_a_vel * dt

                ek = -s_a_angle - s_k_angle
                tk = np.clip((ek * 280.0) - (s_k_vel * 25.0), -400.0, 400.0)
                s_k_vel += ((tk - (knee_mass * g_var * np.sin(s_k_angle))) / knee_mass) * dt
                s_k_angle += s_k_vel * dt

                records.append([sim_id, step, g_var, f_var, target_impulse, s_a_angle, s_k_angle, ea, ek])

            if sim_id % (batch_size // 10 or 1) == 0:
                bar.progress((sim_id + 1) / batch_size)

        cols = ["Sim_ID", "Step", "Gravity", "Friction", "Target", "Ankle_Rad", "Knee_Rad", "Ankle_Err", "Knee_Err"]
        df = pd.DataFrame(records, columns=cols)
        status.text("⚙️ Scaling tensors for machine learning matrices...")
        scaler = StandardScaler()
        df[["Ankle_Rad", "Knee_Rad"]] = scaler.fit_transform(df[["Ankle_Rad", "Knee_Rad"]])

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        bar.progress(1.0)
        status.text("✅ Dataset compiled and normalized!")
        st.metric("Total Datapoints Generated", f"{len(df):,}")
        st.download_button("💾 Download Dataset (.CSV)", csv_data, "optimus_tensor_pack.csv", "text/csv")

