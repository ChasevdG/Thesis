import torch
import numpy as np

class GroupBase(torch.nn.Module):

    def __init__(self, dimension, identity):
        """ Implements a group.

        @param dimension: Dimensionality of the group (number of dimensions in the basis of the algebra).
        @param identity: Identity element of the group.
        """
        super().__init__()
        self.dimension = dimension
        self.register_buffer('identity', torch.Tensor(identity))

    def elements(self):
        """ Obtain a tensor containing all group elements in this group.
        
        """
        raise NotImplementedError()

    def product(self, h, h_prime):
        """ Defines group product on two group elements.

        @param g1: Group element 1
        @param g2: Group element 2
        """
        raise NotImplementedError()

    def inverse(self, h):
        """ Defines inverse for group element.

        @param g: A group element.
        """
        raise NotImplementedError()

    def left_action_on_R2(self, h_batch, x_batch):
        """ Group action of an element from the subgroup H on a vector in R2. For efficiency we
        implement this batchwise.

        @param h_batch: Group elements from H.
        @param x_batch: Vectors in R2.
        """
        raise NotImplementedError()

    def left_action_on_H(self, h_batch, h_prime_batch):
        """ Group action of elements of H on other elements in H itself. Comes down to group product.
        For efficiency we implement this batchwise. Each element in h_batch is applied to each element
        in h_prime_batch.

        @param h_batch: Group elements from H.
        @param h_prime_batch: Other group elements in H.        
        """
        raise NotImplementedError()

    def matrix_representation(self, h):
        """ Obtain a matrix representation in R^2 for an element h.

        @param h: Group element
        """
        raise NotImplementedError()

    def determinant(self, h):
        """ Calculate the determinant of the representation of a group element
        h.

        @param g:
        """
        raise NotImplementedError()
    
    def normalize_group_elements(self, h):
        """ Map the group elements to an interval [-1, 1]. We use this to create
        a standardized input for obtaining weights over the group.

        @param g:
        """
        raise NotImplementedError()


class CyclicGroup(GroupBase):

    def __init__(self, order):
        super().__init__(
            dimension=1,
            identity=[0.]
        )

        assert order > 1
        self.order = torch.tensor(order)

    def elements(self):
        """ Obtain a tensor containing all group elements in this group.
        
        """
        return torch.linspace(
            start=0,
            end=2 * np.pi * float(self.order - 1) / float(self.order),
            steps=self.order,
            device=self.identity.device
        )
    
    def product(self, h1, h2):
        """ Defines group product on two group elements of the cyclic group C4.

        @param h1: Group element 1
        @param h2: Group element 2
        """

        # As we directly parameterize the group by its rotation angles, this 
        # will be a simple addition. Don't forget the closure property though!

        ## YOUR CODE STARTS HERE ##
        product = torch.remainder(h1 + h2, 2 * np.pi)
        ## AND ENDS HERE ##

        return product

    def inverse(self, h):
        """ Defines group inverse for an element of the cyclic group C4.

        @param h: Group element
        """

        # Implement the inverse operation. Keep the closure property in mind!

        ## YOUR CODE STARTS HERE ##
        inverse = torch.remainder(-h, 2 * np.pi)
        ## AND ENDS HERE ##

        return inverse
    
    def left_action_on_R2(self, batch_h, batch_x):
        """ Group action of an element g on a set of vectors in R2.

        @param batch_h: Tensor of group elements. [num_elements]
        @param batch_x: Tensor of vectors in R2. [2, spatial_x, spatial_y]
        """
        # Create a tensor containing representations of each of the group
        # elements in the input. Creates a tensor of size [batch_size, 2, 2].

        ## YOUR CODE STARTS HERE ##
        batched_rep = torch.stack([self.matrix_representation(h) for h in batch_h])
        ## AND ENDS HERE ##

        # Transform the r2 input grid with each representation to end up with
        # a transformed grid of dimensionality [num_group_elements, spatial_x,
        # spatial_y, 2]. Note the order of the dimensions!
        
        # Recall that we are working with a left-regular representation, 
        # meaning we transform vectors in R^2 through left-matrix multiplication.

        ## YOUR CODE STARTS HERE ##
        out = torch.einsum('boi,ixy->bxyo', batched_rep, batch_x)
        ## AND ENDS HERE ## 

        # Afterwards (because grid_sample assummes our grid is y,x instead of x,y) 
        # we swap x and y coordinate values with a roll along final dimension.
        return out.roll(shifts=1, dims=-1)

    def left_action_on_H(self, batch_h, batch_h_prime):
        """ Group action of an element h on a set of group elements in H.
        Nothing more than a batchwise group product.

        @param batch_h: Tensor of group elements.
        @param batch_h_prime: Tensor of group elements to apply group product to.
        """
        # The elements in batch_h work on the elements in batch_h_prime directly,
        # through the group product. Each element in batch_h is applied to each element
        # in batch_h_prime.
        transformed_batch_h = self.product(batch_h.repeat(batch_h_prime.shape[0], 1),
                                           batch_h_prime.unsqueeze(-1))
        return transformed_batch_h

    def matrix_representation(self, h):
        """ Obtain a matrix representation in R^2 for an element h.

        @param h: A group element.
        """
        ## YOUR CODE STARTS HERE ##
        cos_t = torch.cos(h)
        sin_t = torch.sin(h)

        representation = torch.tensor([
            [cos_t, -sin_t],
            [sin_t, cos_t]
        ], device=self.identity.device)
        ## AND ENDS HERE ##

        return representation
    
    def normalize_group_elements(self, h):
        """ Normalize values of group elements to range between -1 and 1.
        The group elements range from 0 to 2pi * (self.order - 1) / self.order,
        so we normalize by 

        @param h: A group element.
        """
        largest_elem = 2 * np.pi * (self.order - 1) / self.order

        return (2*h / largest_elem) - 1.