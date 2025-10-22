import { Tooltip as MuiTooltip, Typography } from "@mui/material";

const Tooltip = ({ children, title, ...props }) => {
  return (
    <MuiTooltip
      arrow={true}
      title={
        typeof title === "string" ? (
          <Typography
            color="special.white"
            style={{ whiteSpace: "pre-line", wordBreak: "break-word" }}
          >
            {title}
          </Typography>
        ) : (
          title
        )
      }
      {...props}
    >
      {children}
    </MuiTooltip>
  );
};

export default Tooltip;
