"""
BLUE-Net: Model-Driven Deep Unfolding Network for Underwater Image Enhancement

This module implements the core BLUE-Net architecture which combines:
1. Physical underwater image formation model: I = J*t + B*(1-t)
2. Deep unrolling/unfolding optimization algorithm
3. Minimum color loss principle for color correction
4. Iterative Proximal Mapping Module (IPMM) for scene radiance refinement

Author: Thuy Thi Pham
"""

# py libs
import torch
import torch.nn as nn

from net import *

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  ## specify the GPU id's, GPU id's start from 0.

# ==============================================================================
# MINIMUM COLOR LOSS PRINCIPLE - Color Correction Module
# ==============================================================================
# The minimum color loss principle addresses color cast in underwater images
# It assumes that in the restored image, the channel with minimum intensity
# should be corrected to match the channel with maximum intensity
# ==============================================================================


def get_mean_value(batch):
    """
    Decompose RGB channels into largest, medium, and smallest based on mean intensity.

    This function implements the first step of the minimum color loss principle:
    identifying which color channels have been most/least attenuated by water.

    Args:
        batch: Input image batch of shape [B, C, H, W] where C=3 (RGB)

    Returns:
        list_mean_sorted: Sorted mean values for each channel [B, 3]
        list_indices: Indices of sorted channels [B, 3]
        largest_channel: Channel with highest mean intensity [B, 1, H, W]
        medium_channel: Channel with medium mean intensity [B, 1, H, W]
        smallest_channel: Channel with lowest mean intensity [B, 1, H, W]
        largest_index: Index of largest channel [B]
        medium_index: Index of medium channel [B]
        smallest_index: Index of smallest channel [B]
    """
    # Get batch size of input
    batch_size = batch.shape[0]

    # Initialize storage for outputs
    list_mean_sorted = []
    list_indices = []
    largest_index = []
    medium_index = []
    smallest_index = []
    largest_channel = []
    medium_channel = []
    smallest_channel = []

    # Process each image in the batch
    for bs in range(batch_size):
        image = batch[bs, :, :, :]  # [C, H, W]

        # Compute mean intensity for each RGB channel
        mean = torch.mean(image, (2, 1))  # [3] - mean for R, G, B

        # Sort channels by mean intensity (ascending order)
        mean_I_sorted, indices = torch.sort(mean)
        list_mean_sorted.append(mean_I_sorted)
        list_indices.append(indices)

        # Extract indices: indices[0]=smallest, indices[1]=medium, indices[2]=largest
        largest_index.append(indices[2])
        medium_index.append(indices[1])
        smallest_index.append(indices[0])

        # Extract the corresponding channel maps
        largest_channel.append(torch.unsqueeze(image[indices[2], :, :], 0))
        medium_channel.append(torch.unsqueeze(image[indices[1], :, :], 0))
        smallest_channel.append(torch.unsqueeze(image[indices[0], :, :], 0))

    # Stack all batch results
    list_mean_sorted = torch.stack(list_mean_sorted)
    list_indices = torch.stack(list_indices)
    largest_index = torch.stack(largest_index)
    medium_index = torch.stack(medium_index)
    smallest_index = torch.stack(smallest_index)
    largest_channel = torch.stack(largest_channel)
    medium_channel = torch.stack(medium_channel)
    smallest_channel = torch.stack(smallest_channel)

    return (
        list_mean_sorted,
        list_indices,
        largest_channel,
        medium_channel,
        smallest_channel,
        largest_index,
        medium_index,
        smallest_index,
    )


def mapping_index(batch, value, index):
    """
    Map corrected channel values back to their original channel positions.

    After color correction, this function reassigns the corrected channel values
    (J_m or J_s) back to their original RGB channel positions.

    Args:
        batch: Image batch [B, C, H, W]
        value: Corrected channel values [B, 1, H, W]
        index: Channel indices indicating where to place the values [B]

    Returns:
        new_batch: Updated image batch with corrected channels [B, C, H, W]
    """
    batch_size = batch.shape[0]
    new_batch = []
    for bs in range(batch_size):
        image = batch[bs, :, :, :]
        # Replace the channel at position index[bs] with the corrected value
        image[index[bs], :, :] = value[bs]
        new_batch.append(image)
    new_batch = torch.stack(new_batch)
    return new_batch


# ==============================================================================
# BASIC BLOCK - One Unrolling Layer of BLUE-Net
# ==============================================================================
# Each BasicBlock represents one iteration of the optimization algorithm.
# It implements ADMM-style updates for all variables: B, t, J, Z, and dual variables.
# ==============================================================================


class BasicBlock(nn.Module):
    """
    One unrolling iteration implementing the following optimization steps:
    1. Color correction via minimum color loss principle
    2. Background light (B) estimation
    3. Transmission map (t) estimation
    4. Scene radiance (J) update
    5. Denoised transmission (Z) via RDN
    6. Dual variable updates (Q, R, u, v, w_1, w_2)
    """

    def __init__(self):
        super(BasicBlock, self).__init__()
        print("Loading subnetworks .....")

        # Z-Net: RDN for denoising transmission map
        # Input: noisy transmission map (3 channels)
        # Output: clean transmission map (3 channels)
        Z_Net = [RDN(3)]
        self.Z_Net = nn.Sequential(*Z_Net)

        # Transmission map channel reduction network
        # Converts 3-channel transmission to 1-channel (grayscale)
        self.t_1D_Net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=False),
            nn.ReLU(),
        )

        # Learnable penalty/regularization parameters (initialized > 1 for stability)
        # These parameters control the trade-off between different terms in optimization
        self.alpha = nn.Parameter(
            torch.tensor([3.001]), requires_grad=True
        )  # Penalty for B-subproblem (unused in forward)
        self.beta = nn.Parameter(
            torch.tensor([3.001]), requires_grad=True
        )  # Penalty for J-subproblem
        self.eta = nn.Parameter(
            torch.tensor([3.001]), requires_grad=True
        )  # Penalty for t-subproblem
        self.lambda_1 = nn.Parameter(
            torch.tensor([1.001]), requires_grad=True
        )  # L1 weight for medium channel
        self.lambda_2 = nn.Parameter(
            torch.tensor([1.001]), requires_grad=True
        )  # L1 weight for smallest channel

    def forward(self, I, t_p, B_p, B, t, J, Y, Z, Q, R, u, v, w_1, w_2, eps=1e-6):
        """
        Forward pass: one unrolling iteration of the optimization algorithm.

        Args:
            I: Observed underwater image [B, 3, H, W]
            t_p: Transmission map prior [B, 3, H, W]
            B_p: Background light prior [B, 3, 1, 1]
            B: Current background light estimate [B, 3, 1, 1]
            t: Current transmission map [B, 3, H, W]
            J: Current scene radiance (clean image) [B, 3, H, W]
            Y: Output from IPMM module [B, 3, H, W]
            Z: Denoised transmission map [B, 3, H, W]
            Q, R: Dual variables for ADMM
            u, v, w_1, w_2: Auxiliary variables for color correction ADMM

        Returns:
            Updated B, t, J, Y, Z, Q, R, u, v, w_1, w_2, beta
        """
        # Extract learnable parameters
        alpha = self.alpha  # Not used in current implementation
        beta = self.beta
        eta = self.eta
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        # Fixed regularization weights for prior terms
        gamma_1 = 0.3  # Weight for background light prior
        gamma_2 = 0.7  # Weight for transmission prior

        # ----------------------------------------------------------------------
        # STEP 1: Minimum Color Loss Principle - Color Correction
        # ----------------------------------------------------------------------
        # Objective: min ||J_l_bar - J_m_bar||_1 + ||J_l_bar - J_s_bar||_1
        # This corrects color cast by aligning medium/small channels with large channel

        # Decompose J into largest, medium, smallest channels
        (
            list_mean_sorted,
            list_indices,
            J_l,
            J_m,
            J_s,
            largest_index,
            medium_index,
            smallest_index,
        ) = get_mean_value(J)

        # Extract mean values and reshape to [B, 1, 1, 1] for broadcasting
        J_l_bar = torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(list_mean_sorted[:, 2], 1), 1), 1
        ).to(DEVICE)
        J_m_bar = torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(list_mean_sorted[:, 1], 1), 1), 1
        ).to(DEVICE)
        J_s_bar = torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(list_mean_sorted[:, 0], 1), 1), 1
        ).to(DEVICE)

        # ADMM updates for auxiliary variables (soft thresholding for L1 norm)
        J_m_bar = J_l - u + (1.0 / lambda_1) * w_1
        J_s_bar = J_l - v + (1.0 / lambda_2) * w_2

        J_l = J_l.to(DEVICE)
        J_m = J_m.to(DEVICE)
        J_s = J_s.to(DEVICE)

        # Apply color correction: shift medium and smallest channels toward largest
        J_m = J_m + torch.mul(J_l_bar - J_m_bar, J_l)
        J_s = J_s + torch.mul(J_l_bar - J_s_bar, J_l)

        # Map corrected channels back to original RGB positions
        J = mapping_index(J.clone(), J_m.clone(), medium_index)
        J = mapping_index(J.clone(), J_s.clone(), smallest_index)

        # Update auxiliary variables u, v (soft-thresholding operator)
        u = torch.sign((J_l_bar - J_m_bar + (1.0 / lambda_1) * w_1)) * F.relu(
            torch.abs((J_l_bar - J_m_bar + (1.0 / lambda_1) * w_1)) - (1.0 / lambda_1),
            inplace=False,
        )
        v = torch.sign((J_l_bar - J_s_bar + (1.0 / lambda_2) * w_2)) * F.relu(
            torch.abs((J_l_bar - J_s_bar + (1.0 / lambda_2) * w_2)) - (1.0 / lambda_2),
            inplace=False,
        )

        # Update dual variables w_1, w_2
        w_1 = w_1 + lambda_1 * (J_l_bar - J_m_bar - u)
        w_2 = w_2 + lambda_2 * (J_l_bar - J_s_bar - v)

        # ----------------------------------------------------------------------
        # STEP 2: Background Light (B) Estimation
        # ----------------------------------------------------------------------
        # Physical model: I = J*t + B*(1-t)
        # Rearrange: B = [I - J*t] / (1-t)
        # Add prior regularization: min ||B - B_p||^2 + data_fidelity
        # Closed-form solution with gamma_1 regularization

        D = torch.ones(I.shape).to(DEVICE)
        # Solve: (1-t)^2 * B + gamma_1 * B = gamma_1 * B_p + (I - J*t)*(1-t)
        B = (gamma_1 * B_p - (J * t - I) * (1 - t)) / ((1.0 - t) * (1 - t) + gamma_1)
        # Average B spatially (assume constant background light)
        B = torch.mean(B, (2, 3), True)  # [B, 3, 1, 1]
        # Broadcast back to image size
        B = B * D

        # ----------------------------------------------------------------------
        # STEP 3: Transmission Map (t) Estimation
        # ----------------------------------------------------------------------
        # Physical model: I = J*t + B*(1-t)
        # Rearrange: t = (I - B) / (J - B)
        # Add prior regularization and Z-constraint
        # Closed-form solution with gamma_2 and eta regularization

        t = (gamma_2 * t_p + eta * Z - R - (B - I) * (J - B)) / (
            (J - B) * (J - B) + gamma_2 + eta
        )
        # Reduce to 1 channel via learned 1x1 conv + ReLU
        t = self.t_1D_Net(t)
        # Replicate to 3 channels for subsequent operations
        t = torch.cat((t, t, t), 1)

        # ----------------------------------------------------------------------
        # STEP 4: Scene Radiance (J) Update
        # ----------------------------------------------------------------------
        # Physical model: I = J*t + B*(1-t)
        # Rearrange: J = [I - B*(1-t)] / t
        # Add IPMM output (Y) as regularization via ADMM
        # Closed-form solution with beta penalty

        J = (beta * Y - Q - (B * (1.0 - t) - I) * t) / (t * t + beta)

        # ----------------------------------------------------------------------
        # STEP 5: Transmission Denoising (Z) via RDN
        # ----------------------------------------------------------------------
        # Apply deep network to denoise transmission map
        Z = self.Z_Net(t + (1.0 / eta) * R)

        # ----------------------------------------------------------------------
        # STEP 6: Dual Variable Updates (ADMM)
        # ----------------------------------------------------------------------
        # Update Lagrange multipliers for constraints
        Q = Q + beta * (J - Y)  # Dual variable for J-Y consistency
        R = R + eta * (t - Z)  # Dual variable for t-Z consistency

        return B, t, J, Y, Z, Q, R, u, v, w_1, w_2, beta


# ==============================================================================
# IPMM - Iterative Proximal Mapping Module
# ==============================================================================
# IPMM acts as a proximal operator for the J-subproblem, replacing simple
# closed-form updates with a deep U-Net that learns to refine scene radiance.
# It uses cross-stage feature fusion (CSFF) for multi-scale processing.
# ==============================================================================


class IPMM(nn.Module):
    """
    Iterative Proximal Mapping Module (Stage 2 of the proximal mapping).

    This module refines the scene radiance J using a U-Net architecture with:
    - Shallow feature extraction with Channel Attention
    - Multi-scale Encoder-Decoder with skip connections
    - Cross-Stage Feature Fusion (CSFF) from previous stage
    - Supervised Attention Module (SAM) for progressive refinement
    """

    def __init__(
        self,
        in_c=3,
        out_c=3,
        n_feat=80,
        scale_unetfeats=48,
        scale_orsnetfeats=32,
        num_cab=8,
        kernel_size=3,
        reduction=4,
        bias=False,
    ):
        super(IPMM, self).__init__()
        act = nn.PReLU()

        # Shallow feature extraction
        self.shallow_feat2 = nn.Sequential(
            conv(3, n_feat, kernel_size, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
        )

        # Multi-scale encoder-decoder
        self.stage2_encoder = Encoder(
            n_feat,
            kernel_size,
            reduction,
            act,
            bias,
            scale_unetfeats,
            depth=4,
            csff=True,
        )
        self.stage2_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4
        )

        # Supervised Attention Module for progressive refinement
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)

        # Learnable interpolation parameter
        self.r1 = nn.Parameter(torch.Tensor([0.5]))

        # Feature concatenation
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)

        # Subspace projection for feature fusion
        self.merge12 = mergeblock(n_feat, 3, True)

    def forward(self, x2_img, stage1_img, feat1, res1, x2_samfeats):
        """
        Forward pass of IPMM.

        Args:
            x2_img: Input image for this stage [B, 3, H, W]
            stage1_img: Output from previous stage [B, 3, H, W]
            feat1: Encoder features from previous stage
            res1: Decoder features from previous stage
            x2_samfeats: SAM features from previous iteration [B, n_feat, H, W]

        Returns:
            x3_samfeats: Updated SAM features for next iteration
            stage2_img: Refined output image
            feat2: Encoder features for next stage
            res2: Decoder features for next stage
        """
        # Extract shallow features
        x2 = self.shallow_feat2(x2_img)

        # Merge with previous SAM features via subspace projection
        x2_cat = self.merge12(x2, x2_samfeats)

        # Encode with cross-stage feature fusion
        feat2, feat_fin2 = self.stage2_encoder(x2_cat, feat1, res1)

        # Decode
        res2 = self.stage2_decoder(feat_fin2, feat2)

        # Generate output via SAM
        x3_samfeats, stage2_img = self.sam23(res2[-1], x2_img)

        return x3_samfeats, stage2_img, feat2, res2


# ==============================================================================
# BLUE_NET - Main Network Architecture
# ==============================================================================
# Combines multiple unrolling layers (BasicBlock) with IPMM for iterative
# refinement of underwater images. Each layer performs one optimization iteration.
# ==============================================================================


class Model(torch.nn.Module):
    """
    BLUE-Net: Balanced Layer-wise Unfolding Enhancement Network.

    Architecture:
    1. Initial IPMM stage (Stage 1)
    2. LayerNo iterations of: BasicBlock (physics-based updates) + IPMM (learning-based refinement)
    3. Progressive refinement of J, B, t through unrolling

    The network outputs enhanced images at each layer for deep supervision.
    """

    def __init__(self, LayerNo=5):
        """
        Initialize BLUE-Net.

        Args:
            LayerNo: Number of unrolling layers (typically 5)
        """
        super(Model, self).__init__()

        self.LayerNo = LayerNo

        # Create LayerNo BasicBlocks (one per unrolling iteration)
        net_layers = []
        for i in range(LayerNo):
            net_layers.append(BasicBlock())
        self.uunet = nn.ModuleList(net_layers)

        # IPMM hyperparameters
        in_c = 3  # Input channels (RGB)
        out_c = 3  # Output channels (RGB)
        n_feat = 40  # Number of features
        scale_unetfeats = 20  # Feature scaling for U-Net encoder/decoder
        scale_orsnetfeats = 16  # (Not used)
        num_cab = 8  # (Not used)
        kernel_size = 3  # Convolution kernel size
        reduction = 4  # Channel reduction factor for attention
        bias = False  # No bias in convolutions
        depth = 5  # (Not used)
        act = nn.PReLU()  # Activation function

        # ----------------------------------------------------------------------
        # Stage 1 IPMM initialization
        # ----------------------------------------------------------------------
        # Shallow feature extraction
        self.shallow_feat1 = nn.Sequential(
            conv(3, n_feat, kernel_size, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
        )

        # Multi-scale encoder-decoder (depth=4)
        self.stage1_encoder = Encoder(
            n_feat,
            kernel_size,
            reduction,
            act,
            bias,
            scale_unetfeats,
            depth=4,
            csff=True,
        )
        self.stage1_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4
        )

        # Supervised Attention Module
        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)

        # Learnable parameters
        self.r1 = nn.Parameter(torch.Tensor([0.5]))

        # Additional modules (not used in current forward pass)
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.merge12 = mergeblock(n_feat, 3, True)

        # Stage 2+ IPMM (reused for all subsequent iterations)
        self.basic = IPMM(
            in_c=3,
            out_c=3,
            n_feat=40,
            scale_unetfeats=20,
            scale_orsnetfeats=16,
            num_cab=8,
            kernel_size=3,
            reduction=4,
            bias=False,
        )

    def forward(self, I, t_p, B_p):
        """
        Forward pass: iteratively refine underwater images through unrolling.

        Args:
            I: Input degraded underwater image [B, 3, H, W]
            t_p: Transmission map prior [B, 1 or 3, H, W]
            B_p: Background light prior [B, 3, H, W]

        Returns:
            list_output: Enhanced images at each layer [LayerNo x [B, 3, H, W]]
            list_B: Background lights at each layer [LayerNo x [B, 3, 1, 1]]
            list_t: Transmission maps at each layer [LayerNo x [B, 3, H, W]]
        """
        bs, _, _, _ = I.shape

        # ----------------------------------------------------------------------
        # Initialize all variables to zeros
        # ----------------------------------------------------------------------
        # Physical variables
        B = torch.zeros((bs, 3, 1, 1)).to(DEVICE)  # Background light
        t = torch.zeros(I.shape).to(DEVICE)  # Transmission map
        J = I.to(DEVICE)  # Scene radiance (init with input)

        # Auxiliary variables
        X = torch.zeros(I.shape).to(DEVICE)  # (Not used)
        Y = torch.zeros(I.shape).to(DEVICE)  # IPMM output
        Z = torch.zeros(I.shape).to(DEVICE)  # Denoised transmission

        # Dual variables for ADMM
        P = torch.zeros(I.shape).to(DEVICE)  # (Not used)
        Q = torch.zeros(I.shape).to(DEVICE)  # Dual for J-Y constraint
        R = torch.zeros(I.shape).to(DEVICE)  # Dual for t-Z constraint

        # Color correction auxiliary variables
        u = torch.zeros((bs, 1, 1, 1)).to(DEVICE)
        v = torch.zeros((bs, 1, 1, 1)).to(DEVICE)
        w_1 = torch.zeros((bs, 1, 1, 1)).to(DEVICE)
        w_2 = torch.zeros((bs, 1, 1, 1)).to(DEVICE)

        # Storage for outputs at each layer
        list_output = []
        list_B = []
        list_t = []

        # ----------------------------------------------------------------------
        # Stage 1 IPMM: Initial proximal mapping
        # ----------------------------------------------------------------------
        # Compute initial estimate of Y using proximal gradient
        beta = torch.tensor([3.001]).to(DEVICE)
        x1_img = J + (1.0 / beta) * Q

        # Pass through Stage 1 IPMM
        x1 = self.shallow_feat1(x1_img)
        feat1, feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1, feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)

        # Update Y with IPMM output
        Y = stage1_img

        # ----------------------------------------------------------------------
        # Iterative Unrolling: LayerNo iterations
        # ----------------------------------------------------------------------
        for j in range(self.LayerNo):
            # BasicBlock: Physics-based variable updates (B, t, J, Z) + dual updates
            [B, t, J, Y, Z, Q, R, u, v, w_1, w_2, beta] = self.uunet[j](
                I, t_p, B_p, B, t, J, Y, Z, Q, R, u, v, w_1, w_2
            )

            # IPMM: Learning-based refinement of J via proximal mapping
            img = J + (1.0 / beta) * Q  # Proximal gradient step
            x2_samfeats, stage1_img, feat1, res1 = self.basic(
                img, stage1_img, feat1, res1, x2_samfeats
            )

            # Update Y for next iteration
            Y = stage1_img

            # Store outputs for this layer
            list_output.append(J)
            list_B.append(B)
            list_t.append(t)

        return list_output, list_B, list_t
