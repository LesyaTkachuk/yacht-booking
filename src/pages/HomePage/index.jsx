import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ROUTES } from "src/navigation/routes";
import { StyledWrapper } from "./styled";
import { Typography, Stack } from "@mui/material";

const HomePage = () => {
  // TODO delete, example of useQuery GET api call
  // const { isPending, error, data, isFetching } = useQuery({
  //   queryKey: ["repoData"],
  //   queryFn: async () => {
  //     const response = await fetch(
  //       "https://api.github.com/repos/TanStack/query"
  //     );
  //     return await response.json();
  //   },
  // });

  // set filters to search params
  return (
    <StyledWrapper>
      <Stack width={"60%"}>
        <Typography variant="subtitle" color={"special.white"}>
          From sailing dreams to unforgettable escapes
        </Typography>
        <Typography variant="h1" color={"special.white"}>
          Choose the Perfect Yacht for Your Voyage
        </Typography>
      </Stack>
    </StyledWrapper>
  );
};

export default HomePage;
