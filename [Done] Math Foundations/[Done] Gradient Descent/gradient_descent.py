class GradientDescent:
    def gradient_descent(
        self,
        iterations: int,
        learning_rate: float,
        init: float,
    ) -> float:
        # Derivative function for f(x) = x^2, which is f'(x) = 2x
        def compute_derivative(value: float) -> float:
            return 2 * value

        # Starting estimate for x
        current_position = float(init)

        # Iteratively take steps in the opposite direction of the gradient
        for _ in range(iterations):
            gradient = compute_derivative(current_position)
            current_position -= learning_rate * gradient

        return round(float(current_position), 5)
