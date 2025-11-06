class CoefficientsAdjuster:
    def __init__(self, coefficient_functions):
        """
        Initialize the class with a list of coefficient adjustment functions.

        :param coefficient_functions: A list of functions, each representing the way to adjust a coefficient based on t.
        """
        assert isinstance(coefficient_functions, dict)
        self.coefficient_functions = coefficient_functions

    def adjust_coefficients(self, t):
        """
        Adjust coefficients based on the input t using the provided functions.

        :param t: The input value t (should be in [0,1]).
        :return: A list of adjusted coefficients.
        """
        # Adjust each coefficient based on its corresponding function and t
        adjusted_coefficients = {
            k: self.coefficient_functions[k](t) for k in self.coefficient_functions.keys()
        }
        return adjusted_coefficients

# Example of how to use the class
if __name__ == '__main__':
    # Define some sample coefficient adjustment functions
    def adjust_coef1(t):
        return 1 + 2 * t  # Adjust first coefficient as a function of t

    def adjust_coef2(t):
        return 3 * (1 - t)  # Adjust second coefficient as a function of t

    def adjust_coef3(t):
        return 0.5 + 0.5 * t  # Adjust third coefficient as a function of t

    # Create a list of these functions
    coefficient_functions = {
        "f1": adjust_coef1,
        "f2": adjust_coef2,
        "f3": adjust_coef3
    }

    # Instantiate the class with the functions
    adjuster = CoefficientsAdjuster(coefficient_functions)

    # Test the adjust_coefficients method
    t = 0.5
    adjusted_coeffs = adjuster.adjust_coefficients(t)
    print(f"Adjusted coefficients at t={t}: {adjusted_coeffs}")
