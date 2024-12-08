import tkinter as tk
from tkinter import Menu, messagebox
import json
from tkinter.filedialog import asksaveasfilename, askopenfilename
import matplotlib.pyplot as plt
from scipy.signal import freqz, lfilter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading

# Definicja współczynników filtra
coefficients = [
    6.22e-19, 0.00065, 0.00124, 0.00154, 0.00122, -1.31e-18, -0.00201, -0.00414,
    -0.00521, -0.00401, 3.09e-18, 0.00605, 0.01187, 0.01432, 0.01063, -5.31e-18,
    -0.01526, -0.02969, -0.03595, -0.02723, 7.09e-18, 0.04401, 0.09742, 0.14883,
    0.18598, 0.19953, 0.18598, 0.14883, 0.09742, 0.04401, 7.09e-18, -0.02723,
    -0.03595, -0.02969, -0.01526, -5.31e-18, 0.01063, 0.01432, 0.01187, 0.00605,
    3.09e-18, -0.00401, -0.00521, -0.00414, -0.00201, -1.31e-18, 0.00122, 0.00154,
    0.00124, 0.00065, 6.22e-19
]

# Definicja domyślnych wartości parametrów
default_params_fir = {
    "f1": 0.275,
    "f2": 0.4,
    "filter_size": 21,
    "transition_bandwidth": 0.01,
    "ripple_passband": 0.001,
    "ripple_stopband": 0.01
}

default_params_pso = {
    "population_size": 150,
    "num_iterations": 200,
    "c1": 2.05,
    "c2": 2.05,
    "omega_max": 1.0,
    "omega_min": 0.4,
    "passband_error_weight": 10.0,  # Wzmocnienie błędu w paśmie przenoszenia
    "stopband_error_weight": 5.0   # Wzmocnienie błędu w paśmie tłumienia
}
advanced_params = default_params_pso.copy()  # Słownik przechowujący domyślne wartości jako dane

stop_search = False

def stop_calculation():
    """Zatrzymuje trwającą pętlę obliczeniową."""
    global stop_search  # Użycie globalnej zmiennej
    stop_search = True
    print("Proces zatrzymany przez użytkownika.")

# Funkcja zapisywania parametrów do pliku JSON
def save_to_file():
    params = {
        "filter_type": filter_type.get(),
        "f1": frequency_entry1.get(),
        "f2": frequency_entry2.get() if frequency_entry2["state"] != "readonly" else None,
        "filter_size": filter_size_entry.get(),
        "transition_bandwidth": passband_entry.get(),
        "ripple_passband": ripple_passband_entry.get(),
        "ripple_stopband": ripple_stopband_entry.get(),
        "pso_params": advanced_params
    }
    filename = asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="Zapisz parametry"
    )
    if filename:
        with open(filename, 'w') as file:
            json.dump(params, file, indent=4)
        messagebox.showinfo("Sukces", f"Parametry zapisano do pliku {filename}.")

# Funkcja wczytywania parametrów z pliku JSON
def open_from_file():
    filename = askopenfilename(
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="Otwórz plik parametrów"
    )
    if filename:
        try:
            with open(filename, 'r') as file:
                params = json.load(file)
            
            # Ustawienie parametrów na podstawie wczytanych danych
            filter_type.set(params.get("filter_type", "Dolnoprzepustowy"))
            frequency_entry1.delete(0, tk.END)
            frequency_entry1.insert(0, params.get("f1", ""))
            if params.get("f2") is not None:
                frequency_entry2.config(state="normal")
                frequency_entry2.delete(0, tk.END)
                frequency_entry2.insert(0, params.get("f2", ""))
                frequency_entry2.config(state="readonly")
            filter_size_entry.delete(0, tk.END)
            filter_size_entry.insert(0, params.get("filter_size", ""))
            passband_entry.delete(0, tk.END)
            passband_entry.insert(0, params.get("transition_bandwidth", ""))
            ripple_passband_entry.delete(0, tk.END)
            ripple_passband_entry.insert(0, params.get("ripple_passband", ""))
            ripple_stopband_entry.delete(0, tk.END)
            ripple_stopband_entry.insert(0, params.get("ripple_stopband", ""))
            
            global advanced_params
            advanced_params.update(params.get("pso_params", {}))

            update_frequency_fields()
            messagebox.showinfo("Sukces", f"Parametry wczytano z pliku {filename}.")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać pliku: {e}")

def calculate():
    global stop_search
    stop_search = False  # Reset flagi przed rozpoczęciem obliczeń

    def calculation_thread():
        global stop_search
        try:
            # Pobierz wartości parametrów z interfejsu
            f1 = float(frequency_entry1.get())
            f2 = float(frequency_entry2.get())
            filter_size = int(filter_size_entry.get())
            transition_bandwidth = float(passband_entry.get())
            ripple_passband = float(ripple_passband_entry.get())
            ripple_stopband = float(ripple_stopband_entry.get())

            # Pobierz dane PSO z globalnego słownika advanced_params
            population_size = int(advanced_params["population_size"])
            iterations = int(advanced_params["num_iterations"])
            c1 = float(advanced_params["c1"])
            c2 = float(advanced_params["c2"])
            w_max = float(advanced_params["omega_max"])
            w_min = float(advanced_params["omega_min"])

            # Liczba współczynników symetrycznych
            sym_coeffs = (filter_size + 1) // 2

            optimal_fitness = float('inf')
            optimal_coeffs = []

            loop_enabled = search_enabled.get()
            search_threshold = float(search_value.get()) if search_value.get() else 150

            if loop_enabled:
                while optimal_fitness > search_threshold:
                    if stop_search:
                        print("Pętla została zatrzymana przez użytkownika.")
                        root.after(0, lambda: messagebox.showinfo("Wymuszono zatrzymanie", "Obliczenia zostały zatrzymane przez użytkownika."))
                        return
                    optimal_coeffs, optimal_fitness = pso_optimize(
                        fitness_func=lambda h: fitness_function(h, f1, f2, transition_bandwidth, ripple_passband, ripple_stopband),
                        sym_coeffs=sym_coeffs,
                        num_coeffs=filter_size,
                        population_size=population_size,
                        iterations=iterations,
                        c1=c1,
                        c2=c2,
                        w_max=w_max,
                        w_min=w_min,
                        f1=f1,
                        f2=f2,
                        transition_bandwidth=transition_bandwidth,
                        ripple_passband=ripple_passband,
                        ripple_stopband=ripple_stopband
                    )
            else:
                optimal_coeffs, optimal_fitness = pso_optimize(
                    fitness_func=lambda h: fitness_function(h, f1, f2, transition_bandwidth, ripple_passband, ripple_stopband),
                    sym_coeffs=sym_coeffs,
                    num_coeffs=filter_size,
                    population_size=population_size,
                    iterations=iterations,
                    c1=c1,
                    c2=c2,
                    w_max=w_max,
                    w_min=w_min,
                    f1=f1,
                    f2=f2,
                    transition_bandwidth=transition_bandwidth,
                    ripple_passband=ripple_passband,
                    ripple_stopband=ripple_stopband
                )

            if stop_search:
                print("Proces został zatrzymany przed rysowaniem wykresów.")
                root.after(0, lambda: messagebox.showinfo("Wymuszono zatrzymanie", "Obliczenia zostały zatrzymane przez użytkownika."))
                return

            # Przekazanie danych do funkcji w głównym wątku
            root.after(0, lambda: display_results(optimal_coeffs, optimal_fitness))

        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Błąd", f"Coś poszło nie tak: {e}"))

    threading.Thread(target=calculation_thread).start()


def display_results(coefficients, fitness):

    messagebox.showinfo("Optymalizacja zakończona", f"Najlepsza wartość funkcji celu: {fitness}")
    """Wyświetla wyniki i rysuje wykresy w głównym wątku."""
    coefficients_field.delete(1.0, tk.END)
    coefficients_field.insert(tk.END, ", ".join(map(str, coefficients)))
    generate_plots(coefficients)

    
def pso_optimize(fitness_func, sym_coeffs, num_coeffs, population_size, iterations, c1, c2, w_max, w_min, f1, f2, transition_bandwidth, ripple_passband, ripple_stopband):
    global stop_search
    # Inicjalizacja populacji cząstek i ich prędkości
    particles = np.random.uniform(-1, 1, (population_size, sym_coeffs))
    velocities = np.random.uniform(-0.01, 0.01, (population_size, sym_coeffs))

    # Inicjalizacja wartości personal best i global best
    personal_best = particles.copy()
    personal_best_fitness = np.array([fitness_func(sym_to_full(p, num_coeffs)) for p in particles])
    global_best = personal_best[np.argmin(personal_best_fitness)]
    global_best_fitness = personal_best_fitness.min()

    # PSO iteracje
    for iteration in range(iterations):
        if stop_search:  # Sprawdzenie, czy zatrzymać obliczenia
            print("PSO zostało zatrzymane.")
            return global_best, global_best_fitness

        # Dynamiczna aktualizacja wagi inercji
        w = w_max - (w_max - w_min) * (iteration / iterations)

        # Dynamiczne wartości c1 i c2
        c1_i = c1 - ((c1 / 2) * iteration / iterations)
        c2_i = (c2 / 2) + ((c2 / 2) * iteration / iterations)

        for i in range(population_size):
            # Losowe współczynniki rand1 i rand2
            r1 = np.random.uniform(0.1, 1)
            r2 = np.random.uniform(0.1, 1)

            # Aktualizacja prędkości cząstki
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best[i] - particles[i])
                + c2 * r2 * (global_best - particles[i])
            )

            # Aktualizacja pozycji cząstki
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], -1, 1)  # Ograniczenie wartości
            # Obliczenie funkcji celu dla zaktualizowanej cząstki
            fitness_value = fitness_func(sym_to_full(particles[i], num_coeffs))

            # Aktualizacja personal best
            if fitness_value < personal_best_fitness[i]:
                personal_best[i] = particles[i]
                personal_best_fitness[i] = fitness_value

        # Aktualizacja global best
        current_best = personal_best[np.argmin(personal_best_fitness)]
        current_best_fitness = personal_best_fitness.min()
        if current_best_fitness < global_best_fitness:
            global_best = current_best
            global_best_fitness = current_best_fitness

        print(f"Iteracja {iteration + 1}/{iterations}, Global best fitness: {global_best_fitness}")

    # Zwrócenie najlepszego rozwiązania
    return sym_to_full(global_best, num_coeffs), global_best_fitness

# Funkcja do uzupełnienia symetrycznych współczynników filtra FIR
def sym_to_full(sym_coeffs, num_coeffs):
    full_coeffs = np.zeros(num_coeffs)
    half = len(sym_coeffs)
    full_coeffs[:half] = sym_coeffs
    full_coeffs[-half:] = sym_coeffs[::-1]
    return full_coeffs


def fitness_function(h,f1,f2,transition_bandwidth, ripple_passband,ripple_stopband):
    
    w, H = freqz(h, worN=512, whole=False)
    
    
        # Determine filter type
    filter_selected = filter_type.get()

    # Compute indices based on filter type
    if filter_selected == "Dolnoprzepustowy":
        # Lowpass
        passband_indices = w <= (f1 - transition_bandwidth / 2) * np.pi
        stopband_indices = w >= (f1 + transition_bandwidth / 2) * np.pi

    elif filter_selected == "Górnoprzepustowy":
        # Highpass
        passband_indices = w >= (f1 + transition_bandwidth / 2) * np.pi
        stopband_indices = w <= (f1 - transition_bandwidth / 2) * np.pi

    elif filter_selected == "Środkowoprzepustowy":
        # Bandpass
        if f2 is None or f2<=f1:
            raise ValueError("Częstotliwość graniczna f2 musi być zdefiniowana dla filtra środkowoprzepustowego.")
        passband_indices = (w >= (f1 + transition_bandwidth / 2) * np.pi) & (w <= (f2 - transition_bandwidth / 2) * np.pi)
        stopband_indices = (w <= (f1 - transition_bandwidth / 2) * np.pi) | (w >= (f2 + transition_bandwidth / 2) * np.pi)

    elif filter_selected == "Środkowozaporowy":
        # Bandstop
        if f2 is None or f2<=f1:
            raise ValueError("Częstotliwość graniczna f2 musi być zdefiniowana dla filtra środkowozaporowego.")
        passband_indices = (w <= (f1 - transition_bandwidth / 2) * np.pi) | (w >= (f2 + transition_bandwidth / 2) * np.pi)
        stopband_indices = (w >= (f1 + transition_bandwidth / 2) * np.pi) & (w <= (f2 - transition_bandwidth / 2) * np.pi)

    else:
        raise ValueError(f"Nieobsługiwany typ filtra: {filter_selected}")

    
    # Idealna odpowiedź
    H_passband = 1
    H_stopband = 0
    
    # Błąd w paśmie przenoszenia
    passband_error = np.sum(np.where(
        np.abs(np.abs(H[passband_indices]) - H_passband) <= ripple_passband,
        0,  # Jeśli różnica mieści się w tolerancji, błąd = 0
        np.abs(np.abs(H[passband_indices]) - H_passband)  # W przeciwnym razie obliczamy błąd
    ))

    # Błąd w paśmie tłumienia
    stopband_error = np.sum(np.where(
        np.abs(np.abs(H[stopband_indices]) - H_stopband) <= ripple_stopband,
        0,  # Jeśli różnica mieści się w tolerancji, błąd = 0
        np.abs(np.abs(H[stopband_indices]) - H_stopband) # W przeciwnym razie obliczamy błąd
    ))
    # #Błąd w paśmie przenoszenia
    # passband_error = np.sum(np.abs(np.abs(np.abs(H[passband_indices]) - H_passband)-ripple_passband))
        
    # #Błąd w paśmie tłumienia
    # stopband_error = np.sum(np.abs(np.abs(np.abs(H[stopband_indices]) - H_stopband)-ripple_stopband)) 
    
    
    
    # Łączna funkcja celu
    J = 10*passband_error + 5*stopband_error
    return J


# Funkcja przywracająca domyślne wartości w głównym oknie
def reset_to_defaults():
    frequency_entry1.delete(0, tk.END)
    frequency_entry1.insert(0, str(default_params_fir["f1"]))
    frequency_entry2.config(state="normal")  # Tymczasowe odblokowanie
    frequency_entry2.delete(0, tk.END)
    frequency_entry2.insert(0, str(default_params_fir["f2"]))
    frequency_entry2.config(state="readonly", readonlybackground="lightgray")
    filter_size_entry.delete(0, tk.END)
    filter_size_entry.insert(0, str(default_params_fir["filter_size"]))
    passband_entry.delete(0, tk.END)
    passband_entry.insert(0, str(default_params_fir["transition_bandwidth"]))
    ripple_passband_entry.delete(0, tk.END)
    ripple_passband_entry.insert(0, str(default_params_fir["ripple_passband"]))
    ripple_stopband_entry.delete(0, tk.END)
    ripple_stopband_entry.insert(0, str(default_params_fir["ripple_stopband"]))

# Funkcja resetu parametrów PSO w ustawieniach zaawansowanych
def reset_pso_params():
    global advanced_params
    advanced_params = default_params_pso.copy()  # Przywrócenie wartości domyślnych
    for param_name, entry in advanced_entries.items():
        entry.delete(0, tk.END)
        entry.insert(0, str(default_params_pso[param_name]))

def display_coefficients():
    coefficients_field.delete(1.0, tk.END)  # Wyczyść pole tekstowe
    coefficients_text = ", ".join([f"{coef}" for coef in coefficients])  # Wyświetl wszystkie współczynniki po przecinku
    coefficients_field.insert(tk.END, coefficients_text)


def open_advanced_settings():
    global advanced_entries  # Zmienna globalna przechowująca pola tekstowe
    advanced_entries = {}

    advanced_window = tk.Toplevel(root)
    advanced_window.title("Ustawienia zaawansowane: Parametry PSO")
    advanced_window.geometry("400x500")
    advanced_window.configure(bg="lightblue")

    # Ramka dla parametrów PSO
    params_frame = tk.Frame(advanced_window, bg="lightblue")
    params_frame.pack(padx=20, pady=10, fill="both", expand=True)

    # Opisy parametrów
    parameters = [
        ("Rozmiar populacji:", "population_size"),
        ("Liczba iteracji:", "num_iterations"),
        ("C1 (współczynnik poznawczy):", "c1"),
        ("C2 (współczynnik społeczny):", "c2"),
        ("Alfa max (inercja maksymalna):", "omega_max"),
        ("Alfa min (inercja minimalna):", "omega_min"),
        ("Wzmocnienie błędu w paśmie przenoszenia:", "passband_error_weight"),
        ("Wzmocnienie błędu w paśmie tłumienia:", "stopband_error_weight")
    ]

    for i, (label_text, param_name) in enumerate(parameters):
        label = tk.Label(params_frame, text=label_text, bg="lightblue", fg="black", anchor="w")
        label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
        entry = tk.Entry(params_frame)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entry.insert(0, str(advanced_params[param_name]))
        advanced_entries[param_name] = entry

    # Przycisk Zapisz i Reset
    buttons_frame = tk.Frame(advanced_window, bg="lightblue")
    buttons_frame.pack(pady=20)

    save_button = tk.Button(buttons_frame, text="Zapisz", bg="dodgerblue", fg="white", font=("Arial", 12, "bold"),
                            command=lambda: save_pso_params(advanced_entries, advanced_window))
    save_button.pack(side="left", padx=10)

    reset_button = tk.Button(buttons_frame, text="Reset", bg="firebrick", fg="white", font=("Arial", 12, "bold"),
                             command=reset_pso_params)
    reset_button.pack(side="left", padx=10)


def save_pso_params(entries,window):
    global advanced_params
    for key, entry in entries.items():
        advanced_params[key] = entry.get()  # Zapisz wartości z pól tekstowych do `advanced_params`
    window.destroy()  # Zamknięcie okna ustawień

def generate_plots(coefficients):
    # Usunięcie poprzednich wykresów, jeśli istnieją
    for frame in plot_frames:
        for widget in frame.winfo_children():
            widget.destroy()
    for widget in impulse_plot_frame.winfo_children():
        widget.destroy()

    # Pobranie częstotliwości granicznych z interfejsu
    try:
        f1 = float(frequency_entry1.get())
        f2 = float(frequency_entry2.get()) if frequency_entry2["state"] == "normal" else None
        # Sprawdzenie, czy częstotliwości są w poprawnym zakresie
        if not (0 <= f1 < 1) or (f2 is not None and not (0 <= f2 < 1)):
            messagebox.showerror("Błąd", "Częstotliwości graniczne muszą być ułamkami Nyquista (w zakresie od 0 do 1).")
            return
    except ValueError:
        messagebox.showerror("Błąd", "Nie można policzyć: niepoprawne częstotliwości graniczne.")
        return

    # Odpowiedź impulsowa
    impulse_response = lfilter(coefficients, 1, [1] + [0] * len(coefficients))
    fig_impulse, ax_impulse = plt.subplots()
    ax_impulse.stem(impulse_response)
    ax_impulse.set_title("Odpowiedź impulsowa filtru")
    ax_impulse.set_xlabel("Próbki")
    ax_impulse.set_ylabel("Amplituda")
    fig_impulse.subplots_adjust(left=0.2, bottom=0.2)
    canvas_impulse = FigureCanvasTkAgg(fig_impulse, master=impulse_plot_frame)
    canvas_impulse.draw()
    canvas_impulse.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Odpowiedź skokowa
    step_response = np.cumsum(impulse_response)
    fig1, ax1 = plt.subplots()
    ax1.plot(step_response)
    ax1.set_title("Odpowiedź skokowa filtru")
    ax1.set_xlabel("Próbki")
    ax1.set_ylabel("Amplituda")
    fig1.subplots_adjust(left=0.2, bottom=0.2)
    canvas1 = FigureCanvasTkAgg(fig1, master=plot_frames[0])
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Charakterystyka amplitudowa z idealnym filtrem
    w, h = freqz(coefficients)
    fig2, ax2 = plt.subplots()
    ax2.plot(w / np.pi, abs(h), label="Filtr zaprojektowany")

    # Generowanie idealnego filtru
    ideal_filter = np.zeros_like(w)
    if filter_type.get() == "Dolnoprzepustowy":
        if f1 is not None:
            ideal_filter[w <= f1 * np.pi] = 1
    elif filter_type.get() == "Górnoprzepustowy":
        if f1 is not None:
            ideal_filter[w >= f1 * np.pi] = 1
    elif filter_type.get() == "Środkowoprzepustowy":
        if f1 is not None and f2 is not None:
            ideal_filter[(w >= f1 * np.pi) & (w <= f2 * np.pi)] = 1
    elif filter_type.get() == "Środkowozaporowy":
        if f1 is not None and f2 is not None:
            ideal_filter[(w < f1 * np.pi) | (w > f2 * np.pi)] = 1

    # Wykres idealnego filtra
    ax2.plot(w / np.pi, ideal_filter, linestyle="--", label="Filtr idealny", color="red")
    ax2.set_title("Charakterystyka amplitudowa filtru")
    ax2.set_xlabel("Częstotliwość (ułamek Nyquista)")
    ax2.set_ylabel("Amplituda")
    ax2.legend()
    fig2.subplots_adjust(left=0.2, bottom=0.2)
    canvas2 = FigureCanvasTkAgg(fig2, master=plot_frames[1])
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Charakterystyka fazowa
    fig3, ax3 = plt.subplots()
    ax3.plot(w / np.pi, np.angle(h))
    ax3.set_title("Charakterystyka fazowa filtru")
    ax3.set_xlabel("Częstotliwość (ułamek Nyquista)")
    ax3.set_ylabel("Faza [radiany]")
    fig3.subplots_adjust(left=0.2, bottom=0.2)
    canvas3 = FigureCanvasTkAgg(fig3, master=plot_frames[2])
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Funkcja do zarządzania stanem pola częstotliwości drugiej granicznej i opisu
def update_frequency_fields():
    if filter_type.get() == "Dolnoprzepustowy":
        frequency_entry2.config(state="readonly", disabledforeground="gray")
        frequency_label2.config(fg="gray")
        frequency_label1.config(fg="black")
        description_label.config(fg="black", text="Częstotliwość graniczna 1: częstotliwość odcięcia powyżej której sygnał jest tłumiony. Wyrażona jest jako ułamek częstotliwości Nyquista.")
    elif filter_type.get() == "Górnoprzepustowy":
        frequency_entry2.config(state="readonly", disabledforeground="gray")
        frequency_label2.config(fg="gray")
        frequency_label1.config(fg="black")
        description_label.config(fg="black", text="Częstotliwość graniczna 1: częstotliwość odcięcia poniżej której sygnał jest tłumiony. Wyrażona jest jako ułamek częstotliwości Nyquista.")
    elif filter_type.get() == "Środkowoprzepustowy":
        frequency_entry2.config(state="normal", disabledforeground="black")
        frequency_label2.config(fg="black")
        frequency_label1.config(fg="black")
        description_label.config(fg="black", text="Częstotliwość graniczna 1 i 2: zakres częstotliwości przepuszczanych. Wyrażone są jako ułamek częstotliwości Nyquista. Częstotliwość 2 powinna być większa od częstotliwości 1.")
    elif filter_type.get() == "Środkowozaporowy":
        frequency_entry2.config(state="normal", disabledforeground="black")
        frequency_label2.config(fg="black")
        frequency_label1.config(fg="black")
        description_label.config(fg="black", text="Częstotliwość graniczna 1 i 2: zakres częstotliwości tłumionych. Wyrażone są jako ułamek częstotliwości Nyquista. Częstotliwość 2 powinna być większa od częstotliwości 1.")


# Pola wprowadzania dla kryteriów filtra
def update_criteria_description(event):
    descriptions = {
        filter_size_entry: "Rozmiar filtra określa liczbę współczynników filtra FIR. Większa liczba daje lepszą precyzję, ale zwiększa złożoność obliczeniową.",
        passband_entry: "Szerokość pasma przejściowego to zakres częstotliwości między pasmem przenoszenia a tłumienia.",
        ripple_passband_entry: "Maksymalne falowanie w paśmie przenoszenia to maksymalna tolerancja dla nierównomierności amplitudy w paśmie przenoszenia.",
        ripple_stopband_entry: "Maksymalne falowanie w paśmie tłumienia określa, jak bardzo sygnał w paśmie tłumienia może być przepuszczany."
    }
    description_label_criteria.config(text=descriptions.get(event.widget, ""))

def on_closing():
    if messagebox.askokcancel("Zamknij", "Czy na pewno chcesz zamknąć aplikację?"):
        root.destroy()
 
# Funkcja zmieniająca stan checkboxa
def update_checkbox_status():
    if search_enabled.get():
        print("Zapętlanie aktywne")
    else:
        print("Zapętlanie wyłączone")
def update_loop_menu():
    advanced_menu.entryconfig(1, label=f"Zapętl: {search_value.get()}")  # Aktualizacja nazwy opcji # Aktualizacja nazwy opcji po zmianie wartości
# Funkcja otwierająca okno "Zapętl"
def open_loop_window():
    
    def save_and_close():
        # Zapisanie wartości i zamknięcie okna
        print(f"Wartość szukana: {search_value.get()}")
        update_loop_menu()
        loop_window.destroy()

    # Tworzenie nowego okna
    loop_window = tk.Toplevel(root)
    loop_window.title("Zapętl - Wartość szukana")
    loop_window.geometry("300x150")
    loop_window.configure(bg="lightblue")

    # Etykieta i pole tekstowe
    label = tk.Label(loop_window, text="Wprowadź wartość szukaną:", bg="lightblue", font=("Arial", 12))
    label.pack(pady=10)

    entry = tk.Entry(loop_window, textvariable=search_value, width=25)
    entry.pack(pady=5)

    # Przycisk OK
    button = tk.Button(loop_window, text="OK", bg="dodgerblue", fg="white", command=save_and_close)
    button.pack(pady=10)        
    
    
# Inicjalizacja głównego okna aplikacji
root = tk.Tk()
root.title("Zastosowanie PSO do projektowania filtrów typu FIR")
root.geometry("1920x1080")
root.configure(bg="midnightblue")  # Kolor obramowania okna

# Przypisanie funkcji do przycisku zamykania okna
root.protocol("WM_DELETE_WINDOW", on_closing)
# Dodanie globalnych zmiennych do zarządzania wyszukiwaniem
search_enabled = tk.BooleanVar(value=False)
search_value = tk.StringVar(value="150")  # Domyślna wartość
# Pasek menu
menu_bar = Menu(root)
plik_menu = Menu(menu_bar, tearoff=0)
plik_menu.add_command(label="Zapisz", command=save_to_file)
plik_menu.add_command(label="Otwórz...", command=open_from_file)
plik_menu.add_separator()
plik_menu.add_command(label="Zamknij", command=root.quit)
menu_bar.add_cascade(label="Plik", menu=plik_menu)
advanced_menu = Menu(menu_bar, tearoff=0)
advanced_menu.add_command(label="Parametry PSO", command=open_advanced_settings)

# Opcja: Zapętl
advanced_menu.add_command(label=f"Zapętl: {search_value.get()}", command=open_loop_window)

# Checkbox na końcu menu
advanced_menu.add_checkbutton(
    label="Włącz zapętlanie",
    variable=search_enabled,
    command=update_checkbox_status
)

# Dodanie menu "Ustawienia zaawansowane" do głównego paska menu
menu_bar.add_cascade(label="Ustawienia zaawansowane", menu=advanced_menu)


root.config(menu=menu_bar)

# Definicja marginesu
margin = 0.03  # 3% odstępu od krawędzi oraz między ramkami

# Ramka dla parametrów (lewa sekcja)
params_frame = tk.Frame(root, bg="gray85")
params_frame.place(relx=margin, rely=margin, relwidth=0.5-3*margin/2, relheight=0.5-3*margin/2)

params_label = tk.Label(params_frame, text="PARAMETRY FIR", bg="royalblue4", fg="white", font=("Arial", 16, "bold"))
params_label.pack(fill=tk.X, pady=margin)

# Ramka kontener dla bloku wyboru typu filtra i częstotliwości granicznych
filter_and_frequency_frame = tk.Frame(params_frame, bg="gray85")
filter_and_frequency_frame.pack(pady=10, padx=10, fill="x")

# Ramka wyboru typu filtra (lewa część)
filter_type_frame = tk.Frame(filter_and_frequency_frame, bg="royalblue4", pady=10, padx=10)
filter_type_frame.pack(side="left", fill="both", expand=True)

filter_type_label = tk.Label(filter_type_frame, text="TYP FILTRA", bg="royalblue4", fg="white", font=("Arial", 12, "bold"))
filter_type_label.pack(anchor="w")

# Typ filtra jako przyciski radiowe
filter_type = tk.StringVar(value="Dolnoprzepustowy")
filter_options = ["Dolnoprzepustowy", "Górnoprzepustowy", "Środkowoprzepustowy", "Środkowozaporowy"]

for option in filter_options:
    rb = tk.Radiobutton(filter_type_frame, text=option, variable=filter_type, value=option, command=update_frequency_fields, bg="royalblue4", fg="white", selectcolor="dodgerblue", anchor="w")
    rb.pack(anchor="w")

# Ramka na częstotliwości graniczne (prawa część)
frequency_frame = tk.Frame(filter_and_frequency_frame, bg="deepskyblue", pady=10, padx=10)
frequency_frame.pack(side="right", fill="both", expand=True)

# Pole do wprowadzania częstotliwości granicznych w jednej linii
frequency_label1 = tk.Label(frequency_frame, text="Częstotliwość graniczna 1:", bg="deepskyblue", fg="white")
frequency_label1.grid(row=0, column=0, padx=5, pady=5, sticky="e")
frequency_entry1 = tk.Entry(frequency_frame)
frequency_entry1.grid(row=0, column=1, padx=5, pady=5)

frequency_label2 = tk.Label(frequency_frame, text="Częstotliwość graniczna 2:", bg="deepskyblue", fg="gray")
frequency_label2.grid(row=0, column=2, padx=5, pady=5, sticky="e")
frequency_entry2 = tk.Entry(frequency_frame, state="readonly", disabledforeground="gray")
frequency_entry2.grid(row=0, column=3, padx=5, pady=5)

# Etykieta opisu dynamicznego
description_label = tk.Label(frequency_frame, text="", bg="deepskyblue", fg="white", wraplength=300, justify="left")
description_label.grid(row=1, column=0, columnspan=4, padx=5, pady=10, sticky="w")

# Ramka dla kryteriów filtra (analogicznie do filter_and_frequency_frame, z podziałem góra-dół)
criteria_frame = tk.Frame(params_frame, bg="gray85")
criteria_frame.pack(pady=10, padx=10, fill="x")

# Górna część criteria_frame z tytułem
criteria_label = tk.Label(criteria_frame, text="KRYTERIA FILTRA", bg="royalblue4", fg="white", font=("Arial", 12, "bold"))
criteria_label.pack(fill="x", pady=0)

# Ramka zawierająca pola wprowadzania i dynamiczny opis
criteria_content_frame = tk.Frame(criteria_frame, bg="deepskyblue", pady=10, padx=10)
criteria_content_frame.pack(fill="both", expand=True)

criteria_fields_frame = tk.Frame(criteria_content_frame, bg="deepskyblue")
criteria_fields_frame.grid(row=0, column=0, sticky="w")

description_label_criteria = tk.Label(criteria_content_frame, text="", bg="deepskyblue", fg="black", wraplength=300, justify="left")
description_label_criteria.grid(row=0, column=1, padx=20, sticky="n")

filter_size_label = tk.Label(criteria_fields_frame, text="Rozmiar filtra (liczba współczynników):", bg="deepskyblue", fg="black")
filter_size_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
filter_size_entry = tk.Entry(criteria_fields_frame)
filter_size_entry.grid(row=0, column=1, padx=10, pady=5)
filter_size_entry.bind("<FocusIn>", update_criteria_description)

passband_label = tk.Label(criteria_fields_frame, text="Szerokość pasma przejściowego:", bg="deepskyblue", fg="black")
passband_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
passband_entry = tk.Entry(criteria_fields_frame)
passband_entry.grid(row=1, column=1, padx=10, pady=5)
passband_entry.bind("<FocusIn>", update_criteria_description)

ripple_passband_label = tk.Label(criteria_fields_frame, text="Maksymalne falowanie w paśmie przenoszenia:", bg="deepskyblue", fg="black")
ripple_passband_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
ripple_passband_entry = tk.Entry(criteria_fields_frame)
ripple_passband_entry.grid(row=2, column=1, padx=10, pady=5)
ripple_passband_entry.bind("<FocusIn>", update_criteria_description)

ripple_stopband_label = tk.Label(criteria_fields_frame, text="Maksymalne falowanie w paśmie tłumienia:", bg="deepskyblue", fg="black")
ripple_stopband_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")
ripple_stopband_entry = tk.Entry(criteria_fields_frame)
ripple_stopband_entry.grid(row=3, column=1, padx=10, pady=5)
ripple_stopband_entry.bind("<FocusIn>", update_criteria_description)

# Przycisk Oblicz i Reset
buttons_frame = tk.Frame(params_frame, bg="gray85")
buttons_frame.pack(pady=20)  # Użycie pack zamiast grid

calculate_button = tk.Button(buttons_frame, text="Oblicz", font=("Arial", 12, "bold"), bg="dodgerblue", fg="white", command=calculate)
calculate_button.pack(side="left", padx=10)

reset_button = tk.Button(buttons_frame, text="Reset", font=("Arial", 12, "bold"), bg="sienna1", fg="white", command=reset_to_defaults)
reset_button.pack(side="left", padx=10)
# Przycisk Stop
stop_button = tk.Button(buttons_frame, text="Stop", font=("Arial", 12, "bold"), bg="firebrick", fg="white", command=stop_calculation)
stop_button.pack(side="left", padx=10)
results_frame = tk.Frame(root, bg="gray85")
results_frame.place(relx=margin, rely=0.5+margin/2, relwidth=0.5-3*margin/2, relheight=0.5-3*margin/2)

# Nagłówek ramki "results_frame"
results_label = tk.Label(results_frame, text="UZYSKANY FILTR", bg="royalblue4", fg="white", font=("Arial", 16, "bold"))
results_label.pack(fill=tk.X, pady=margin)

# Wykres odpowiedzi impulsowej w nowej ramce
impulse_plot_frame = tk.Frame(results_frame, bg="white", relief="solid", bd=1)
impulse_plot_frame.place(relx=0.05, rely=margin+0.06, relwidth=0.9, relheight=0.5)

# Pole tekstowe dla współczynników filtra
coefficients_label = tk.Label(results_frame, text="WYPIS WSPÓŁCZYNNIKÓW FILTRA", bg="royalblue4", fg="white", font=("Arial", 12, "bold"))
coefficients_label.place(relx=0.05, rely=0.65, relwidth=0.9)

coefficients_field = tk.Text(results_frame, wrap=tk.WORD, bg="white", fg="black", height=5, font=("Arial", 10))
coefficients_field.place(relx=0.05, rely=0.7, relwidth=0.9, relheight=0.2)

# Ramka dla wykresów (prawa sekcja)
charts_frame = tk.Frame(root, bg="gray85")
charts_frame.place(relx=0.5 + margin / 2, rely=margin, relwidth=0.5-3*margin/2, relheight=1 - 2*margin)

charts_label = tk.Label(charts_frame, text="UZYSKANE WYKRESY", bg="royalblue4", fg="white", font=("Arial", 16, "bold"))
charts_label.pack(fill=tk.X, pady=margin)

# Lista do przechowywania ramek na wykresy
plot_frames = []

# Wysokość, jaką zajmuje tytuł "UZYSKANE WYKRESY"
title_height = 0.025  
usable_height = 1 - 2 * margin - title_height  # Całkowita dostępna wysokość po odjęciu marginesów i tytułu
gap = 0.02  # Odstęp między wykresami
num_plots = 3  # Liczba wykresów
plot_height = (usable_height - gap * (num_plots - 1)) / num_plots  # Wysokość pojedynczego wykresu

# Tworzenie miejsc na wykresy
for i in range(num_plots):  # Trzy miejsca na wykresy
    # Ramka dla miejsca na wykres z przerwami
    plot_frame = tk.Frame(charts_frame, bg="white", relief="solid", bd=1)
    plot_frame.place(
        relx=0.05,  # Odstęp od lewej
        rely=title_height + margin + i * (plot_height + gap),  # Uwzględnienie tytułu, marginesu i odstępu
        relwidth=0.9,  # 90% szerokości charts_frame
        relheight=plot_height  # Obliczona wysokość wykresu
    )
    plot_frames.append(plot_frame)  # Dodanie ramki do listy plot_frames



# Domyślne ustawienia przy starcie
reset_to_defaults()
# Wywołanie funkcji `update_frequency_fields` po ustawieniu elementów interfejsu
update_frequency_fields()

# Uruchomienie głównej pętli aplikacji
root.mainloop()
