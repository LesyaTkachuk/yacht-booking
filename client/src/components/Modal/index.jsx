import { Box, Stack, Typography } from "@mui/material";

import { Content, StyledModal } from "./styled";

const Modal = ({ children, title, subtitle, open, onClose, actions }) => (
  <StyledModal open={open} onClose={onClose} disableEnforceFocus>
    <Content alignCenter={"align-center"} width={"600px"}>
      <Stack mb={4} pr={4} gap={1}>
        {title && <Typography variant="h5">{title}</Typography>}
        {subtitle && (
          <Typography variant="body1" color="secondary">
            {subtitle}
          </Typography>
        )}
      </Stack>

      <Stack
        width="100%"
        overflow="auto"
        pr={4}
        justifyContent={"space-between"}
        alignItems={"center"}
      >
        {children}
        {actions && (
          <Box width="100%" pt={2} pr={4}>
            {actions}
          </Box>
        )}
      </Stack>
    </Content>
  </StyledModal>
);

export default Modal;
