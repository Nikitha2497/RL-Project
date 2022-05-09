import numpy as np

#This is the same one used in the assignments
#This represents the base class of a policy

class Policy(object):
    def action_prob(self,state,action:int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        raise NotImplementedError()

    def action(self,state) -> int:
        """
        input:
            state
        return:
            action
        """
        raise NotImplementedError()

    def print_all(self):
        """
        print all the action prob
        """
        raise NotImplementedError()

    def save_tofile(self, filename):
        """
        save the state to a file
        """
        raise NotImplementedError()

    def restore_fromfile(self, filename):
        """
        restores the policy from file
        """
        raise NotImplementedError()


