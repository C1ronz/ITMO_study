import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm  
import warnings
warnings.filterwarnings('ignore')

# ==================== ЧАСТЬ 1: ПОИСК ЯДРА ====================
def find_kernel(matrix):
    """
    Находит базис ядра матрицы (нуль-пространства)
    """
    matrix = np.array(matrix, dtype=float)
    U, S, Vh = np.linalg.svd(matrix)
    rank = np.sum(S > 1e-10)
    kernel_basis = Vh[rank:].T
    return kernel_basis

# ==================== ЧАСТЬ 2: АЛГОРИТМ ГРАМА-ШМИДТА ====================
def gram_schmidt(vectors):
    """
    Классический алгоритм ортогонализации Грама-Шмидта
    """
    vectors = np.array(vectors, dtype=float)
    n = len(vectors)
    orthogonal = []
    
    for i in range(n):
        v = vectors[i].copy()
        
        for u in orthogonal:
            v = v - np.dot(v, u) / np.dot(u, u) * u
        
        if np.linalg.norm(v) > 1e-10:  
            orthogonal.append(v)
    
    # Нормализация
    orthonormal = [v / np.linalg.norm(v) for v in orthogonal]
    
    return np.array(orthogonal), np.array(orthonormal)

# ==================== ЧАСТЬ 3: ТЕОРЕМА ГАМИЛЬТОНА-КЭЛИ (ИСПРАВЛЕНА) ====================
def hamilton_cayley_theorem(matrix):
    """
    Проверка теоремы Гамильтона-Кэли: характеристический многочлен
    аннулирует матрицу: p(A) = 0
    """
    A = np.array(matrix, dtype=float) 
    n = A.shape[0]
    
 
    coeffs = np.poly(A)  
    coeffs = coeffs.astype(float)
    

    result = np.zeros_like(A)
    for i, coeff in enumerate(coeffs):
        power = n - i
        if power == 0:
            result += coeff * np.eye(n, dtype=float)
        else:
            result += coeff * np.linalg.matrix_power(A, power)
    
    # Проверяем, является ли результат нулевой матрицей
    is_zero = np.allclose(result, np.zeros_like(A), atol=1e-10)
    
    return is_zero, result, coeffs

# ==================== ЧАСТЬ 4: ВЫЧИСЛЕНИЕ e^(At) ====================
def matrix_exponential(A, t):
    """
    Вычисление матричной экспоненты e^(At)
    """
    return expm(np.array(A, dtype=float) * t)

def matrix_exponential_series(A, t, terms=20):
    """
    Вычисление e^(At) через ряд Тейлора (для демонстрации)
    """
    A = np.array(A, dtype=float)
    At = A * t
    result = np.eye(len(A), dtype=float)
    term = np.eye(len(A), dtype=float)
    
    for k in range(1, terms):
        term = term @ At / k
        result += term
    
    return result

# ==================== ВИЗУАЛИЗАЦИЯ ====================
class Visualizer3D:
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 6))
        
    def create_cube(self, center=(0,0,0), size=1):
        """
        Создание вершин и ребер куба
        """
        s = size / 2
        vertices = np.array([
            [-s, -s, -s], [ s, -s, -s], [ s, -s,  s], [-s, -s,  s],  
            [-s,  s, -s], [ s,  s, -s], [ s,  s,  s], [-s,  s,  s]  
        ])
        vertices += center
        
    
        edges = [
            (0,1), (1,2), (2,3), (3,0),  
            (4,5), (5,6), (6,7), (7,4), 
            (0,4), (1,5), (2,6), (3,7)  
        ]
        
        return vertices, edges
    
    def apply_operator(self, vertices, operator):
        """
        Применение линейного оператора к вершинам куба
        """
        return vertices @ operator.T
    
    def plot_cube(self, vertices, edges, ax, color='blue', alpha=0.3, label=''):
        """
        Отрисовка куба
        """

        for edge in edges:
            points = vertices[list(edge)]
            ax.plot3D(points[:,0], points[:,1], points[:,2], 
                     color=color, linewidth=2, alpha=alpha)
        

        ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], 
                  color=color, s=50, alpha=alpha)
        
    def plot_eigenvectors(self, eigenvectors, eigenvalues, ax, center=(0,0,0)):
        """
        Отображение собственных векторов
        """
        colors = ['red', 'green', 'purple', 'orange', 'brown']
        max_len = max(np.linalg.norm(v) for v in eigenvectors)
        
        for i, (vec, val) in enumerate(zip(eigenvectors, eigenvalues)):

            direction = vec / np.linalg.norm(vec)
            length = 2 * abs(val) if abs(val) > 0.1 else 1.5
            arrow = direction * length
            
            ax.quiver(center[0], center[1], center[2],
                     arrow[0], arrow[1], arrow[2],
                     color=colors[i % len(colors)], 
                     linewidth=3, 
                     alpha=0.8,
                     arrow_length_ratio=0.2,
                     label=f'λ={val:.2f}')
    
    def visualize_operator_action(self, operator, title="Действие линейного оператора"):
        """
        Визуализация действия оператора на куб
        """

        ax1 = self.fig.add_subplot(121, projection='3d')
        ax2 = self.fig.add_subplot(122, projection='3d')
        

        vertices, edges = self.create_cube(center=(0,0,0), size=1)
        self.plot_cube(vertices, edges, ax1, color='blue', alpha=0.7, label='Исходный')
        ax1.set_title('Исходный куб')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        

        transformed_vertices = self.apply_operator(vertices, operator)
        self.plot_cube(transformed_vertices, edges, ax2, color='red', alpha=0.7, label='Преобразованный')
        ax2.set_title('Куб после действия оператора')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        

        for ax in [ax1, ax2]:
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            ax.set_zlim([-3, 3])
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_with_eigenvectors(self, operator):
        """
        Визуализация оператора вместе с собственными векторами
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        

        vertices, edges = self.create_cube(center=(0,0,0), size=1)
        transformed_vertices = self.apply_operator(vertices, operator)
        self.plot_cube(transformed_vertices, edges, ax, color='lightblue', alpha=0.5)
        
   
        eigenvalues, eigenvectors = np.linalg.eig(operator)
        self.plot_eigenvectors(eigenvectors.T, eigenvalues, ax)
        
        ax.set_title('Действие оператора и собственные векторы')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_zlim([-4, 4])
        ax.legend()
        plt.show()

# ==================== ДЕМОНСТРАЦИЯ ====================
def demonstrate_all():
    """
    Демонстрация всех реализованных функций
    """
    print("="*60)
    print("ДЕМОНСТРАЦИЯ")
    print("="*60)
    
    # Создаем тестовую матрицу 3x3
    A = np.array([
        [2, 1, 0],
        [0, 2, 1],
        [0, 0, 2]
    ])
    
    print("\n1. ПОИСК ЯДРА МАТРИЦЫ:")
    print("-" * 40)
    kernel = find_kernel(A)
    print(f"Матрица A:\n{A}")
    print(f"\nБазис ядра:\n{kernel}")
    print(f"Размерность ядра: {kernel.shape[1] if kernel.size > 0 else 0}")
    
    print("\n2. АЛГОРИТМ ГРАМА-ШМИДТА:")
    print("-" * 40)
    vectors = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    print(f"Исходные векторы:\n{vectors}")
    orthogonal, orthonormal = gram_schmidt(vectors)
    print(f"\nОртогональные векторы:\n{orthogonal}")
    print(f"\nОртонормированные векторы:\n{orthonormal}")
    
    # Проверка ортогональности
    print(f"\nСкалярные произведения ортонормированных векторов:")
    for i in range(len(orthonormal)):
        for j in range(i+1, len(orthonormal)):
            dot = np.dot(orthonormal[i], orthonormal[j])
            print(f"  v{i+1}·v{j+1} = {dot:.10f}")
    
    print("\n3. ТЕОРЕМА ГАМИЛЬТОНА-КЭЛИ:")
    print("-" * 40)
    is_valid, result, coeffs = hamilton_cayley_theorem(A)
    print(f"Матрица A:\n{A}")
    print(f"\nХарактеристический многочлен:")
    print(f"  p(λ) = ", end="")
    n = len(coeffs) - 1
    for i, coeff in enumerate(coeffs):
        if abs(coeff) > 1e-10:
            if i > 0:
                print(" + " if coeff > 0 else " - ", end="")
            print(f"{abs(coeff):.2f}·λ^{n-i}", end="")
    print()
    print(f"\nПроверка p(A) = 0: {is_valid}")
    if not is_valid:
        print(f"Максимальное отклонение: {np.max(np.abs(result)):.2e}")
    
    print("\n4. МАТРИЧНАЯ ЭКСПОНЕНТА e^(At):")
    print("-" * 40)
    t_values = [0, 0.5, 1.0]
    for t in t_values:
        exp_At = matrix_exponential(A, t)
        print(f"\nПри t = {t}:")
        print(f"e^(A·{t}) =\n{exp_At.round(4)}")
    
    print("\n5. ВИЗУАЛИЗАЦИЯ:")
    print("-" * 40)
    
    # Создаем визуализатор
    viz = Visualizer3D()
    
    # Визуализируем действие оператора
    print("\n→ Визуализация действия оператора на куб...")
    viz.visualize_operator_action(A, "Действие оператора A на единичный куб")
    
    # Визуализируем с собственными векторами
    print("\n→ Визуализация с собственными векторами...")
    viz.visualize_with_eigenvectors(A)

    
    return A, viz

demonstrate_all()
