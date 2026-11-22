import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Autoencoder-Pretrained Image Classifier", layout="wide")

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


@st.cache_data
def load_data():
    (xtr, ytr), (xte, yte) = keras.datasets.fashion_mnist.load_data()
    xtr = xtr.astype("float32")[..., None] / 255.0
    xte = xte.astype("float32")[..., None] / 255.0
    return xtr, ytr, xte, yte


def build_encoder():
    inp = keras.Input((28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    z = layers.Dense(64, activation="relu")(x)
    return keras.Model(inp, z, name="encoder")


def build_decoder():
    inp = keras.Input((64,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(inp)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D()(x)
    out = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
    return keras.Model(inp, out, name="decoder")


def build_classifier(encoder):
    return keras.Sequential([encoder, layers.Dropout(0.3),
                             layers.Dense(10, activation="softmax")])


def plot_history(h1, h2):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot(h1.history["loss"], label="train")
    ax[0].plot(h1.history["val_loss"], label="val")
    ax[0].set_title("Autoencoder loss (MSE)")
    ax[0].legend()
    ax[1].plot(h2.history["accuracy"], label="train")
    ax[1].plot(h2.history["val_accuracy"], label="val")
    ax[1].set_title("Fine-tune accuracy")
    ax[1].legend()
    fig.tight_layout()
    return fig


def plot_recon(orig, recon):
    fig, ax = plt.subplots(2, 8, figsize=(10, 3))
    for i in range(8):
        ax[0, i].imshow(orig[i].squeeze(), cmap="gray")
        ax[1, i].imshow(recon[i].squeeze(), cmap="gray")
        ax[0, i].axis("off")
        ax[1, i].axis("off")
    ax[0, 0].set_title("Original", fontsize=8, loc="left")
    ax[1, 0].set_title("Reconstructed", fontsize=8, loc="left")
    return fig


def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=90, fontsize=7)
    ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
    fig.tight_layout()
    return fig


st.title("Autoencoder-Pretrained Image Classifier")
st.caption("Semi-supervised learning: Autoencoder (unlabeled) + Fine-tuning (few labels) | Fashion-MNIST")

xtr, ytr, xte, yte = load_data()

with st.sidebar:
    st.header("Settings")
    n_labeled = st.select_slider("Labeled images", [100, 250, 500, 1000, 2000], value=500)
    ae_epochs = st.slider("Autoencoder epochs (unlabeled)", 1, 15, 8)
    ft_epochs = st.slider("Fine-tune epochs (labeled)", 5, 50, 30)
    train_btn = st.button("Train", type="primary")

if train_btn:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(xtr), n_labeled, replace=False)
    x_lab, y_lab = xtr[idx], ytr[idx]

    with st.spinner("Step 1: Autoencoder pretraining (labels use nahi ho rahe)..."):
        enc, dec = build_encoder(), build_decoder()
        ae = keras.Model(enc.input, dec(enc.output))
        ae.compile(optimizer="adam", loss="mse")
        h1 = ae.fit(xtr, xtr, epochs=ae_epochs, batch_size=256,
                    validation_data=(xte, xte), verbose=0)
        recon = ae.predict(xte[:8], verbose=0)

    with st.spinner("Step 2: Fine-tuning on labeled data..."):
        clf = build_classifier(enc)
        clf.compile(optimizer=keras.optimizers.Adam(1e-3),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h2 = clf.fit(x_lab, y_lab, epochs=ft_epochs, batch_size=16,
                     validation_data=(xte, yte), verbose=0)

    with st.spinner("Step 3: Baseline (no pretraining) for comparison..."):
        base = build_classifier(build_encoder())
        base.compile(optimizer=keras.optimizers.Adam(1e-3),
                     loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        base.fit(x_lab, y_lab, epochs=ft_epochs, batch_size=16, verbose=0)

    pre_acc = clf.evaluate(xte, yte, verbose=0)[1]
    base_acc = base.evaluate(xte, yte, verbose=0)[1]
    y_pred = np.argmax(clf.predict(xte, verbose=0), axis=1)

    st.session_state["res"] = dict(clf=clf, pre=pre_acc, base=base_acc, h1=h1, h2=h2,
                                   recon=recon, y_pred=y_pred, n=n_labeled)

res = st.session_state.get("res")
if not res:
    st.info("Sidebar se settings choose karo aur **Train** dabao.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("With AE pretraining", f"{res['pre']*100:.2f}%")
c2.metric("Without pretraining (baseline)", f"{res['base']*100:.2f}%")
c3.metric("Improvement", f"{(res['pre']-res['base'])*100:+.2f}%")
st.caption(f"Only {res['n']} labeled images use hui hain; autoencoder ne 60,000 unlabeled images se features seekhe.")

t1, t2, t3, t4 = st.tabs(["Training curves", "Reconstruction", "Confusion matrix", "Try it"])
with t1:
    st.pyplot(plot_history(res["h1"], res["h2"]))
with t2:
    st.pyplot(plot_recon(xte[:8], res["recon"]))
with t3:
    st.pyplot(plot_cm(yte, res["y_pred"]))
with t4:
    left, right = st.columns(2)
    with left:
        st.subheader("Test set image")
        i = st.number_input("Index (0-9999)", 0, 9999, 0)
        p = res["clf"].predict(xte[i:i + 1], verbose=0)[0]
        st.image(xte[i].squeeze(), width=150, clamp=True)
        st.write(f"Actual: **{CLASSES[yte[i]]}** | Predicted: **{CLASSES[int(p.argmax())]}** ({p.max()*100:.1f}%)")
    with right:
        st.subheader("Upload your own image")
        up = st.file_uploader("PNG/JPG", type=["png", "jpg", "jpeg"])
        inv = st.checkbox("Invert colors (white background wali image ke liye)", value=True)
        if up:
            img = ImageOps.grayscale(Image.open(up)).resize((28, 28))
            arr = np.array(img, dtype="float32") / 255.0
            if inv:
                arr = 1.0 - arr
            p = res["clf"].predict(arr[None, ..., None], verbose=0)[0]
            st.image(arr, width=150, clamp=True)
            st.write(f"Prediction: **{CLASSES[int(p.argmax())]}** ({p.max()*100:.1f}%)")
            st.bar_chart(dict(zip(CLASSES, p.tolist())))
