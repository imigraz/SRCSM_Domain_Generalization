import numpy as np
import random
from generators.generator_base import GeneratorBase


class ImageSpacingGenerator(GeneratorBase):
    """
    Generator that uses an sitk image (or a list of sitk images) and an sitk transformation as an input and calculates an output size.
    """
    def __init__(self,
                 dim,
                 use_z_spacing_close_to_isotropic=False,
                 spacing_range_in_plane=None,
                 random_spacing_in_range=False,
                 # reverse_order=True,
                 # output_size,
                 # output_spacing=None,
                 # valid_output_sizes=None,
                 *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param output_size: The resampled output image size in sitk format ([x, y] or [x, y, z]). May contain entries
                            that are None. In this case, the corresponding dimension will either take the smallest value
                            out of valid_output_sizes in which the resampled image fits. If valid_output_sizes is not
                            defined, the output size will be calculated, such that resampled image fits exactly in the
                            output image.
        :param output_spacing: The resampled output spacing.
        :param valid_output_sizes: A list of valid output sizes per dimension (a list of lists). See output_size
                                   parameter for usage.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(ImageSpacingGenerator, self).__init__(*args, **kwargs)
        self.dim = dim
        self.use_z_spacing_close_to_isotropic = use_z_spacing_close_to_isotropic
        self.spacing_range_in_plane = spacing_range_in_plane
        self.random_spacing_in_range = random_spacing_in_range
        # self.reverse_order = reverse_order
        # self.output_size = output_size
        # self.output_spacing = output_spacing or [1] * dim
        # self.valid_output_sizes = valid_output_sizes

    def get(self, image):
        """
        Calculates the output size for the given sitk image or an extent based on the parameters.
        :param image: The sitk image.
        :param extent: The extent.
        :return: The output size as a list.
        """
        assert (image is not None)
        output_spacing = [x for x in list(image.GetSpacing())]
        spacing_x = output_spacing[0]
        spacing_y = output_spacing[1]
        spacing_z = output_spacing[2]

        if self.spacing_range_in_plane is not None:
            min_in_plane, max_in_plane = self.spacing_range_in_plane
            if self.random_spacing_in_range:
                rand_spacing = random.uniform(min_in_plane, max_in_plane)
                spacing_x = rand_spacing
                spacing_y = rand_spacing

            else:
                if spacing_x > max_in_plane:
                    spacing_x = (max_in_plane + min_in_plane) / 2.0  # use average of min and max spacing
                if spacing_x < min_in_plane:
                    spacing_x = min_in_plane  # use min to preserve higher resolution

                if spacing_y > max_in_plane:
                    spacing_y = (max_in_plane + min_in_plane) / 2.0  # use average of min and max spacing
                if spacing_y < min_in_plane:
                    spacing_y = min_in_plane  # use min to preserve higher resolution

        if self.use_z_spacing_close_to_isotropic:
            target_spacing = (spacing_x + spacing_y) / 2.0
            integer_dividend = round(spacing_z / target_spacing)
            spacing_z = spacing_z / integer_dividend
        output_spacing = [spacing_x, spacing_y, spacing_z]

        # output_spacing = [1, 2, 3]

        # if self.reverse_order:
        #     output_spacing = list(reversed(output_spacing))
        return output_spacing

        # output_size = []
        # assert (image is not None) != (extent is not None) , 'Either image or extent must be set.'
        # if extent is None:
        #     extent = [image.GetSize()[i] * image.GetSpacing()[i] for i in range(self.dim)]
        # for i in range(self.dim):
        #     if self.output_size[i] is not None:
        #         # if the output size is fixed for the current dimension, use it.
        #         output_size.append(self.output_size[i])
        #     elif self.valid_output_sizes is not None and self.valid_output_sizes[i] is not None:
        #         # if the output size is None, but valid_output_sizes is not None,
        #         # use minimal valid_output_sizes such that the resampled image fits.
        #         size = int(np.ceil(extent[i] / self.output_spacing[i]))
        #         valid_size = None
        #         for valid_size in sorted(self.valid_output_sizes[i]):
        #             if size < valid_size:
        #                 # break as soon as the image fits into the current size
        #                 break
        #         output_size.append(valid_size)
        #     else:
        #         # otherwise (output_size is None and valid_output_sizes is None), calculate the
        #         # output size such that the resampled image fits exactly
        #         size = int(np.ceil(extent[i] / self.output_spacing[i]))
        #         output_size.append(size)
        # return output_size
