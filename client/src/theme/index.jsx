import { createTheme, alpha, darken } from "@mui/material/styles";
import { inputClasses } from "@mui/material";
import { palette } from "./palette";

export const theme = createTheme({
  palette,
  spacing: 4,
  fontFamily: "Inter, sans-serif",
  typography: {
    h1: {
      fontFamily: "Montserrat",
      fontSize: "72px",
      fontWeight: "800",
      lineHeight: "80px",
    },
    h2: {
      fontFamily: "Montserrat",
      fontSize: "48px",
      fontWeight: "700",
      lineHeight: "56px",
    },
    h3: {
      fontFamily: "Montserrat",
      fontSize: "40px",
      fontWeight: "500",
      lineHeight: "48px",
    },
    h4: {
      fontFamily: "Montserrat",
      fontSize: "32px",
      fontWeight: "800",
      lineHeight: "40px",
    },
    h5: {
      fontFamily: "Montserrat",
      fontSize: "28px",
      fontWeight: "500",
      lineHeight: "34px",
    },
    subtitle: {
      fontFamily: "Inter",
      fontSize: "24px",
      fontWeight: "500",
      lineHeight: "32px",
    },
    subtitle2: {
      fontFamily: "Inter",
      fontSize: "20px",
      fontWeight: "500",
      lineHeight: "26px",
    },
  },
  components: {
    MuiTypography: {
      styleOverrides: {
        root: {
          color: "#000000",
          textAlign: "center",
        },
      },
    },
    MuiDivider: {
      styleOverrides: {
        root: `
          border-color: ${palette.secondary.main};
        `,
      },
    },
    MuiSvgIcon: {
      styleOverrides: {
        fontSizeSmall: {
          width: 14,
          height: 14,
        },
        fontSizeMedium: {
          width: 24,
          height: 24,
        },
        fontSizeLarge: {
          width: 28,
          height: 28,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          input: {
            color: palette.primary.dark,
          },
          "&&:hover": {
            "& fieldset": {
              border: `1px solid ${palette.secondary.main}`,
            },
          },
          [`.${inputClasses.focused}`]: {
            "& fieldset": {
              border: `2px solid ${palette.primary.main} !important`,
              backgroundColor: alpha(palette.primary.main, 0.08),
            },
          },
          "& fieldset": {
            borderRadius: 12,
            border: `1px solid ${palette.secondary.dark}`,
          },
          [`.${inputClasses.error}`]: {
            "& fieldset": {
              border: `1px solid ${palette.error.main} !important`,
            },
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          textTransform: "none",
          boxShadow: "none",
          gap: 10,
        },
        contained: {
          minWidth: "100px",
          color: palette.special.white,
          "&:hover": {
            boxShadow: "none",
            backgroundColor: darken(palette.primary.main, 0.4),
          },
        },
        outlined: {
          borderWidth: "2px !important",
          "&:hover": {
            boxShadow: "none",
            borderColor: darken(palette.primary.main, 0.4),
            color: darken(palette.primary.main, 0.4),
          },
        },
        text: {
          padding: 0,
          color: palette.primary.light,
        },
      },
    },
  },
});
