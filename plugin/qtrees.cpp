/******************************************************************************
 * INCLUDES
 ******************************************************************************/

#include "OsApi.h"
#include "EventLib.h"
#include "LuaEngine.h"
#include "QtreesClassifier.h"

/******************************************************************************
 * DEFINES
 ******************************************************************************/

#define LUA_QTREES_LIBNAME    "qtrees"

/******************************************************************************
 * LOCAL FUNCTIONS
 ******************************************************************************/

/*----------------------------------------------------------------------------
 * qtrees_version
 *----------------------------------------------------------------------------*/
int qtrees_version (lua_State* L)
{
    lua_pushstring(L, BINID);
    lua_pushstring(L, BUILDINFO);
    return 2;
}

/*----------------------------------------------------------------------------
 * qtrees_open
 *----------------------------------------------------------------------------*/
int qtrees_open (lua_State *L)
{
    static const struct luaL_Reg qtrees_functions[] = {
        {"version",             qtrees_version},
        {"classifier",          QtreesClassifier::luaCreate},
        {NULL,                  NULL}
    };

    /* Set Library */
    luaL_newlib(L, qtrees_functions);

    return 1;
}

/******************************************************************************
 * EXPORTED FUNCTIONS
 ******************************************************************************/

extern "C" {
void initqtrees (void)
{
    /* Initialize Modules */
    QtreesClassifier::init();

    /* Extend Lua */
    LuaEngine::extend(LUA_QTREES_LIBNAME, qtrees_open);

    /* Indicate Presence of Package */
    LuaEngine::indicate(LUA_QTREES_LIBNAME, BINID);

    /* Display Status */
    print2term("%s plugin initialized (%s)\n", LUA_QTREES_LIBNAME, BINID);
}

void deinitqtrees (void)
{
}
}
