import os
import json

STATE_PATH = "storage/project.json"

DEFAULT_STATE = {
    "classes": {},
    "model_trained": False
}


def load_state():
    os.makedirs("storage", exist_ok=True)

    if not os.path.exists(STATE_PATH):
        save_state(DEFAULT_STATE)
        return DEFAULT_STATE

    try:
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
            if isinstance(state.get("classes"), list):
                state["classes"] = {}
            return state
    except:
        save_state(DEFAULT_STATE)
        return DEFAULT_STATE


def save_state(state):
    os.makedirs("storage", exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=4)


# ✅ IMPORTANT: THIS FIXES YOUR ERROR
def add_class(class_name):
    state = load_state()

    if class_name not in state["classes"]:
        state["classes"][class_name] = []

    save_state(state)
    return state


def add_samples(class_name, image_path):
    state = load_state()

    if class_name not in state["classes"]:
        state["classes"][class_name] = []

    state["classes"][class_name].append(image_path)

    save_state(state)
    return state


def set_trained(value=True):
    state = load_state()
    state["model_trained"] = value
    save_state(state)
    return state

def remove_class(class_name):
    state = load_state()
    if class_name in state["classes"]:
        del state["classes"][class_name]
    save_state(state)
    return state