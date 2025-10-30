"use client";

import { alpha, IconButton, MenuItem, styled, Typography } from "@mui/material";

import TextField from "../TextField";

export const StyledTextField = styled(TextField)`
  & .MuiTypography-root {
    text-align: left;
    color: ${({ theme }) => theme.palette.primary.main};
  }
`;

export const HiddenMenuItem = styled(MenuItem)`
  display: none;
`;

export const StyledIconButton = styled(IconButton)`
  &:hover {
    background-color: transparent;
  }
`;

export const StyledTypography = styled(Typography)`
  overflow: hidden;
  text-overflow: ellipsis;
`;

export const StyledMenuItem = styled(MenuItem)`
  padding: 10px 20px;
  transition: background-color 0.2s ease-in;
  border-bottom: 1px solid ${({ theme }) => theme.palette.special.lightGrey};

  &:hover {
    background-color: ${({ theme }) => alpha(theme.palette.info.light, 0.2)};
  }

  &.Mui-selected:hover {
    background-color: ${({ theme }) => alpha(theme.palette.info.light, 0.06)};
  }
`;
